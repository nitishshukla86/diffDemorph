import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import argparse
from pipeline import DDPMPipeline
from utils import get_transforms

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Demorph a face using DDPMPipeline")
parser.add_argument("--img_path", type=str, required=True, help="Path to the input morphed image")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the output image")
parser.add_argument("--num_steps", type=int, default=10, help="Number of diffusion steps")
parser.add_argument("--img_size", type=int, default=256, help="Size of generated images")
args = parser.parse_args()

# -----------------------------
# Config & transforms
# -----------------------------
SEED = 0


#download the model
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="nitishshukla/pretrained-weights",
    allow_patterns=["diffDemorph/*"],
    local_dir="./pretrained"
)


transform,invTrans=get_transforms(image_size=args.img_size)
def postprocess(tensor):
    tensor = tensor.clamp(-1, 1)
    # tensor = (tensor + 1) / 2  # scale to [0,1]
    return tensor

# -----------------------------
# Load pipeline
# -----------------------------
pipeline = DDPMPipeline.from_pretrained(
    './pretrained/diffDemorph'
).to('cuda')

# -----------------------------
# Load input image
# -----------------------------
img = cv2.imread(args.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = transform(img).unsqueeze(0).cuda()  # [1,3,H,W]

# -----------------------------
# Generate demorphed outputs
# -----------------------------
with torch.no_grad():
    outputs = pipeline(
        batch_size=1,
        morph_emb=img_tensor,
        generator=torch.Generator(device='cpu').manual_seed(SEED),
        num_steps=args.num_steps
    )

fake1, fake2 = outputs.chunk(2, dim=1)  # split channels
fake1 = postprocess(fake1[0]).cpu()
fake2 = postprocess(fake2[0]).cpu()
morph_img = invTrans(img_tensor[0].cpu())

# -----------------------------
# Display and save with titles
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, img, title in zip(axes, [morph_img, fake1, fake2], ["MORPH", "OUT1", "OUT2"]):
    ax.imshow(invTrans(img) if isinstance(img, torch.Tensor) else img)
    ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.tight_layout()
plt.savefig(args.save_path)
plt.close()

print(f"Saved demorphed result to {args.save_path}")
