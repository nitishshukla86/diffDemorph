# models.py
from diffusers import UNet2DModel

def get_unet_model(config):
    """
    Returns a UNet2DModel configured for DDPMPipeline.
    
    Args:
        config: An object with at least `latent_size` attribute.
    
    Returns:
        UNet2DModel instance
    """
    model = UNet2DModel(
        sample_size=config.latent_size,      # target image resolution
        in_channels=9,                       # number of input channels
        out_channels=6,                      # number of output channels
        layers_per_block=2,                  # ResNet layers per block
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )
    )
    return model
