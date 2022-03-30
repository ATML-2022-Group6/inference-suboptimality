from functools import partial

from jax import nn, random, lax
from jax import numpy as jnp
from jax.example_libraries import stax

from utils import HyperParams, log_normal

def build_aux_flow(hps: HyperParams):
  """Normalizing flow + auxiliary variable."""
  num_flows: int = 2
  hidden_size: int = hps.flow_hidden_size
  latent_size: int = hps.latent_size
  image_size: int = hps.image_size

  # Nets for `_norm_flow()` method; eq. (9) and (10) in Cremer et al.
  # We assume that each net for each flow assumes the same architecture.
  flow_net_init, flow_net = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # h
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size), # μ
      stax.Dense(latent_size), # logit
    ),
  )

  # Nets for `sample()` method.
  # These nets are for the q(v|x, z) model and the reverse model r(v|x,z).
  # We assume that q(v|x, z) and r(v|x, z) share the same architecture.
  model_net_init, model_net = stax.serial(
    stax.Dense(hidden_size), stax.Elu,
    stax.Dense(hidden_size), stax.Elu,
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size),
      stax.Dense(latent_size),
    ),
  )

  def init_fun(rng):
    rngs = random.split(rng, num=6)

    # Flow procedure.
    _, flow1_net1_params = flow_net_init(rngs[0], input_shape=(image_size+latent_size,))
    _, flow1_net2_params = flow_net_init(rngs[1], input_shape=(image_size+latent_size,))
    _, flow2_net1_params = flow_net_init(rngs[2], input_shape=(image_size+latent_size,))
    _, flow2_net2_params = flow_net_init(rngs[3], input_shape=(image_size+latent_size,))

    # Auxiliary variable procedure.
    _, qv_net_params = model_net_init(rngs[4], input_shape=(image_size+latent_size,))
    _, rv_net_params = model_net_init(rngs[5], input_shape=(image_size+latent_size,))

    params = (
      (
        (flow1_net1_params, flow1_net2_params),
        (flow2_net1_params, flow2_net2_params),
      ),
      (
        qv_net_params,
        rv_net_params,
      ),
    )

    return params

  def sample(rng, z0, x, params):
    """The forward function essentially."""
    norm_flow_params, aux_var_params = params
    qv_net_params, rv_net_params = aux_var_params

    # Auxiliary variable step; forward model q(v|x,z).
    zx = jnp.concatenate((z0, x))
    v0, logqv0 = _forward_aux_var(rng, zx, qv_net_params)

    # Flow procedure.
    logdet = jnp.zeros(shape=(num_flows, latent_size))
    _flow_proc = partial(_flow_proc_full, norm_flow_params=norm_flow_params)
    (zT, vT, _, _), logdet = lax.scan(_flow_proc, (z0, v0, x, 0), logdet)
    inverse_logdet_sum = -jnp.sum(logdet, axis=0)

    # Auxiliary variable step; reverse model r(v|x,z).
    zx = jnp.concatenate((zT, x))
    logrvT = _reverse_aux_var(zx, vT, rv_net_params)

    # Auxiliary flow correction to log q(z|x).
    logprob = logqv0 + inverse_logdet_sum - logrvT

    return zT, logprob
  
  def _flow_proc_full(carry, logdet, norm_flow_params):
    last_zT, last_vT, x, flow_idx = carry
    zT, vT, logdet_flow = _norm_flow(last_zT, last_vT, x,
                                norm_flow_params[flow_idx])
    logdet = logdet + logdet_flow
    flow_idx += 1
    return (zT, vT, x, flow_idx), logdet
  
  def _norm_flow(z, v, x, params):
    """
    Real-NVP (Dinh et al.) normalizing flow as defined
    by equation (9) and (10) in our paper Cremer et al.
    """
    net1_params, net2_params = params

    # Our flow's affine coupling procedure (modulo indexing).
    #
    #               z
    #             /   \
    #          z_1    z_2
    #          /       |
    #        h_1       |
    #       /   \      |
    #      μ_1   σ_1   |
    #       |     |    |
    #      z_2•σ_1 + μ_1 -> z_2' (eq. 9)
    #                        |
    #                       h_2
    #                      /   \
    #                    μ_2   σ_2
    #                     |     |
    #                z_1•σ_2 + μ_2 -> z_1' (eq. 10)
    #

    # Equation (9) in paper. No difference doing z_1 or z_2 first in the coupling.
    zx = jnp.concatenate((z, x))
    μ1, logit_1 = flow_net(net1_params, zx)
    σ1 = nn.sigmoid(logit_1)
    v = v*σ1 + μ1

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    vx = jnp.concatenate((v, x))
    μ2, logit_2 = flow_net(net2_params, vx)
    σ2 = nn.sigmoid(logit_2)
    z = z*σ2 + μ2

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet

  def _forward_aux_var(rng, zx, net_params):
    """Forward model q(v|z,x) in auxiliary variable procedure."""
    mean_v0, logvar_v0 = model_net(net_params, zx)
    eps = random.normal(rng, mean_v0.shape)
    v0 = mean_v0 + eps*jnp.exp(0.5 * logvar_v0)
    logqv0 = log_normal(v0, mean_v0, logvar_v0)
    return v0, logqv0

  def _reverse_aux_var(zx, v, net_params):
    """Reverse model r(v|z,x) in auxiliary variable procedure."""
    mean_v, logvar_v = model_net(net_params, zx)
    logrv = log_normal(v, mean_v, logvar_v)
    return logrv

  return init_fun, sample