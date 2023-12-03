from diffusers import UNet2DModel
import typing as tp

def get_model(repo_id: tp.Optional[str] = None, **kwargs) -> UNet2DModel:
    model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True) if repo_id else UNet2DModel(
            in_channels=3,  
            out_channels=3,
            layers_per_block=1, 
            block_out_channels=(64, 64, 128), 
            down_block_types=(
                "DownBlock2D",  
                "AttnDownBlock2D",  
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  
                "UpBlock2D",
            ),
            **kwargs
        )
    
    return model