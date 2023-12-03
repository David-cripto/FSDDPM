from diffusers import UNet2DModel
import typing as tp

def get_model(repo_id: tp.Optional[str] = None, **kwargs) -> UNet2DModel:
    model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True) if repo_id else UNet2DModel(
            in_channels=3,  
            out_channels=3,
            **kwargs
        )
    
    return model