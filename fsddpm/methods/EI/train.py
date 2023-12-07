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

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps
    
    def reverse(self, score_fn, probability_flow=False):
        T = self.T
        sde_fn = self.sde

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

        return RSDE()

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
    pred_noise = model(perturbed_x, random_t)["sample"]
    loss = torch.mean(torch.sum((pred_noise - z)**2, dim=(1,2,3)))
    return loss

