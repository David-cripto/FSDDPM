import torch
import numpy as np
import jax
import jax.numpy as jnp

class VPSDE():
    def __init__(self, beta_min=0.1, beta_max=20):
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        N: number of discretization steps
        """
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.t2alpha_fn = lambda t: jnp.exp(2 * (-0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0))
        log_alpha_fn = lambda t: jnp.log(self.t2alpha_fn(t))
        grad_log_alpha_fn = jax.grad(log_alpha_fn)
        self.d_log_alpha_dtau_fn = jax.vmap(grad_log_alpha_fn)

    @property
    def T(self):
        return 1

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def psi(self, t_start, t_end):
        return jnp.sqrt(self.t2alpha_fn(t_end) / self.t2alpha_fn(t_start))

    def eps_integrand(self, vec_t):
        d_log_alpha_dtau = self.d_log_alpha_dtau_fn(vec_t)
        integrand = -0.5 * d_log_alpha_dtau / jnp.sqrt(1 - self.t2alpha_fn(vec_t))
        return integrand

def loss_fn(sde, model, x, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t)
    perturbed_x = mean + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)["sample"]
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    return loss

