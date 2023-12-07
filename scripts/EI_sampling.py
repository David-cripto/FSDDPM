from fsddpm.methods.EI.sampling import sample
from fsddpm.methods.EI.train import VPSDE
from fsddpm.methods.model import get_model
import torch

vpsde = VPSDE()
ts_order = 2.0
num_step=10
ab_order=3
B, C, H, W = 1, 3, 28, 28
TIME_EMB_TYPE = "fourier"

model = get_model(sample_size = H, time_embedding_type = TIME_EMB_TYPE)
def eps_fn(x_t, scalar_t):
    vec_t = (torch.ones(x_t.shape[0])).float().to(x_t) * scalar_t
    with torch.no_grad():
        score = model(x_t, vec_t)
    std = vpsde.marginal_prob(torch.zeros_like(score), vec_t)[1]
    eps = - score / std[:, None, None, None]
    return eps


# Example how to sample image
def main():

    image = sample(
        sde = vpsde, 
        eps_fn = eps_fn, 
        ts_order=ts_order, 
        num_step=num_step, 
        ab_order=ab_order, 
        noise=torch.randn(B, C, H, W)
    )

if __name__ == '__main__':
    main()