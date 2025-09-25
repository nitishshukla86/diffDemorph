import os
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder

from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Local modules
from dataset import SMDD
from pipeline import DDPMPipeline
from utils import get_transforms
from models import get_unet_model


def get_training_args():
    parser = argparse.ArgumentParser(description="DDPM Training Configuration")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=256)
    parser.add_argument("--dataset_csv_path", type=str, default="self_made_morphs.csv")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="smdd-image-no-condition")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def evaluate(config, epoch, pipeline, batch, invTrans, device):
    """
    Evaluate model outputs and save concatenated images.
    """
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)

    for ind in range(config.eval_batch_size):
        morph = batch['morphed_image'][ind].unsqueeze(0).to(device)
        img1 = batch['img1'][ind].unsqueeze(0).to(device)
        img2 = batch['img2'][ind].unsqueeze(0).to(device)

        # Generate outputs
        with torch.no_grad():
            images = pipeline(
                batch_size=1,
                morph_emb=morph,
                generator=torch.Generator(device='cpu').manual_seed(config.seed)
            )

        # Split pipeline outputs
        fake1, fake2 = images.chunk(2, 1)

        # Concatenate for comparison: MORPH | OUT1 | OUT2 | IMG1 | IMG2
        concat = torch.cat([morph.cpu(), fake1.cpu(), fake2.cpu(), img1.cpu(), img2.cpu()], dim=-1)

        # Save as PIL image
        invTrans(concat[0]).save(f"{test_dir}/{epoch:04d}_{ind}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, val_dataloader, lr_scheduler, invTrans):
    """
    Main training loop with accelerator and gradient accumulation.
    """
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    device = accelerator.device
    torch.manual_seed(config.seed)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name,
                exist_ok=True,
                private=config.hub_private_repo
            ).repo_id
        accelerator.init_trackers("DDPM_train")

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            morph = batch['morphed_image'].to(device)
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)

            clean_images = torch.cat([img1, img2], dim=1)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],),
                device=device, dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noisy_images = torch.cat([noisy_images, morph], dim=1)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Evaluation & save
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                val_batch = next(iter(val_dataloader))
                evaluate(config, epoch, pipeline, val_batch, invTrans, device)

                pipeline.save_pretrained(os.path.join(config.output_dir, f"{epoch:04d}"))
                if config.push_to_hub:
                    upload_folder(
                        repo_id=config.hub_model_id,
                        folder_path=os.path.join(config.output_dir, f"{epoch:04d}"),
                        commit_message=f"Upload epoch {epoch:04d}"
                    )


if __name__ == "__main__":
    config = get_training_args()
    transform, invTrans = get_transforms(image_size=config.image_size)

    trainset = SMDD(train=True, transform=transform, path=config.dataset_csv_path)
    testset = SMDD(train=False, transform=transform, path=config.dataset_csv_path)

    train_loader = DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(testset, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, num_workers=2)

    model = get_unet_model(config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=len(train_loader) * config.num_epochs
    )

    train_loop(config, model, noise_scheduler, optimizer, train_loader, val_loader, lr_scheduler, invTrans)
