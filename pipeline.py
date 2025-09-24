import torch
from typing import Optional, Union, Tuple
from diffusers import DiffusionPipeline, ImagePipelineOutput
from PIL import Image
import warnings

class DDPMPipeline(DiffusionPipeline):

    def __init__(self, unet, scheduler):
        super().__init__()
        # scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        morph_emb=None,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        num_steps=50,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        all_images=[]
        image = torch.randn(
            (batch_size, self.unet.in_channels-3, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_steps)

        for t in self.scheduler.timesteps:
            image=torch.cat([image,morph_emb],1)
            model_output = self.unet(image, t).sample
            image = self.scheduler.step(torch.cat([model_output,morph_emb],1), t, image, generator=generator).prev_sample
            image=torch.cat([image.split(3,1)[0],image.split(3,1)[1]],1)
            all_images.append(image)
        return image