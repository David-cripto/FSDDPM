from fsddpm.methods.EI.sampling import sample
from fsddpm.methods.EI.train import VPSDE
from fsddpm.methods.model import get_model
import torch
import jax.numpy as jnp

vpsde = VPSDE()
ts_order = 2.0
num_step=10
ab_order=3
B, C, H, W = 1, 1, 28, 28
TIME_EMB_TYPE = "fourier"

model = get_model(sample_size = H, time_embedding_type = TIME_EMB_TYPE)

# Example how to sample image
def main():
    with torch.no_grad():
        image = sample(
            sde = vpsde, 
            eps_fn = model, 
            ts_order=ts_order, 
            num_step=num_step, 
            ab_order=ab_order, 
            noise=jnp.asarray(torch.randn(B, C, H, W))
        )

if __name__ == '__main__':
    main()