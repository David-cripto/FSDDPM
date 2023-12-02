from diffusers import UNet2DModel
import typing as tp

def get_model(img_size, repo_id: tp.Optional[str] = None) -> UNet2DModel:
    model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True) if repo_id else UNet2DModel(
            sample_size=img_size,  
            in_channels=3,  
            out_channels=3
        )
    
    return model