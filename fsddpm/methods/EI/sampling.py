import jax
import jax.numpy as jnp
import numpy as np
import torch

def get_integrator_basis_fn(sde):
    def _worker(t_start, t_end, num_item):
        dt = (t_end - t_start) / num_item

        t_inter = jnp.linspace(t_start, t_end, num_item, endpoint=False)
        psi_coef = sde.psi(t_inter, t_end)
        integrand = sde.eps_integrand(t_inter)

        return psi_coef * integrand, t_inter, dt
    return _worker


def single_poly_coef(t_val, ts_poly, coef_idx=0):
    num = t_val - ts_poly
    denum = ts_poly[coef_idx] - ts_poly
    num = num.at[coef_idx].set(1.0)
    denum = denum.at[coef_idx].set(1.0)
    return jnp.prod(num) / jnp.prod(denum)

vec_poly_coef = jax.vmap(single_poly_coef, (0, None, None), 0)


def get_one_coef_per_step_fn(sde):
    _eps_coef_worker_fn = get_integrator_basis_fn(sde)
    def _worker(t_start, t_end, ts_poly, coef_idx=0,num_item=10000):
        integrand, t_inter, dt = _eps_coef_worker_fn(t_start, t_end, num_item)
        poly_coef = vec_poly_coef(t_inter, ts_poly, coef_idx)
        return jnp.sum(integrand * poly_coef) * dt
    return _worker

def get_coef_per_step_fn(sde, highest_order, order):
    eps_coef_fn = get_one_coef_per_step_fn(sde)
    def _worker(t_start, t_end, ts_poly, num_item=10000):
        rtn = jnp.zeros((highest_order+1, ), dtype=float)
        ts_poly = ts_poly[:order+1]
        coef = jax.vmap(eps_coef_fn, (None, None, None, 0, None))(t_start, t_end, ts_poly, jnp.flip(jnp.arange(order+1)), num_item)
        rtn = rtn.at[:order+1].set(coef)
        return rtn
    return _worker

def get_ab_eps_coef_order0(sde, highest_order, timesteps):
    _worker = get_coef_per_step_fn(sde, highest_order, 0)
    col_idx = jnp.arange(len(timesteps)-1)[:,None]
    idx = col_idx + jnp.arange(1)[None, :]
    vec_ts_poly = timesteps[idx]
    return jax.vmap(
        _worker,
        (0, 0, 0), 0
    )(timesteps[:-1], timesteps[1:], vec_ts_poly)

def get_ab_eps_coef(sde, highest_order, timesteps, order):
    if order == 0:
        return get_ab_eps_coef_order0(sde, highest_order, timesteps)
    
    prev_coef = get_ab_eps_coef(sde, highest_order, timesteps[:order+1], order=order-1)

    cur_coef_worker = get_coef_per_step_fn(sde, highest_order, order)

    col_idx = jnp.arange(len(timesteps)-order-1)[:,None]
    idx = col_idx + jnp.arange(order+1)[None, :]
    vec_ts_poly = timesteps[idx]
    

    cur_coef = jax.vmap(
        cur_coef_worker,
        (0, 0, 0), 0
    )(timesteps[order:-1], timesteps[order+1:], vec_ts_poly) #[3, 4, (0,1,2,3)]

    return jnp.concatenate(
        [
            prev_coef,
            cur_coef
        ],
        axis=0
    )

def ab_step(x, ei_coef, new_eps, eps_pred):
    x_coef, eps_coef = ei_coef[0], ei_coef[1:]
    full_eps_pred = [new_eps, *eps_pred]
    rtn = x_coef * x
    for cur_coef, cur_eps in zip(eps_coef, full_eps_pred):
        rtn += cur_coef * cur_eps
    return rtn, full_eps_pred[:-1]


def get_rev_ts(exp_sde, num_step, ts_order):
    t0, t1 = 0, exp_sde.T
    t0 = t0 + 1e-1
    rev_ts = jnp.power(
        jnp.linspace(
            jnp.power(t1, 1.0 / ts_order),
            jnp.power(t0, 1.0 / ts_order),
            num_step + 1
        ),
        ts_order
    )
    return rev_ts

def jax2th(array, th_array=None):
    if th_array is None:
        return torch.from_numpy(
            np.asarray(array).copy()
        )
    else:
        return torch.from_numpy(
            np.asarray(array).copy()
        ).to(th_array.device)

def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val

def sample(sde, eps_fn, ts_order, num_step, ab_order, noise):
    rev_ts = get_rev_ts(sde, num_step, ts_order)
    
    x_coef = sde.psi(rev_ts[:-1], rev_ts[1:])
    eps_coef = get_ab_eps_coef(sde, ab_order, rev_ts, ab_order)
    ab_coef = jnp.concatenate([x_coef[:, None], eps_coef], axis=1)
    rev_ts, ab_coef = jax2th(rev_ts), jax2th(ab_coef)
    rev_ts, ab_coef = rev_ts.to(noise.device), ab_coef.to(noise.device)
    
    def ab_body_fn(i, val):
        x, eps_pred = val
        s_t= rev_ts[i]
        
        new_eps = eps_fn(x, s_t)
        new_x, new_eps_pred = ab_step(x, ab_coef[i], new_eps, eps_pred)
        return new_x, new_eps_pred


    eps_pred = [noise,] * ab_order
    img, _ = fori_loop(0, num_step, ab_body_fn, (noise, eps_pred))
    return img