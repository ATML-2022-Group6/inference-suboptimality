import jax
from jax import nn, random
from jax import numpy as jnp
from jax.example_libraries import stax

from utils import HyperParams, log_normal

def build_aux_flow(hps: HyperParams):
  """Normalizing flow + auxiliary variable."""
  num_flows: int = hps.num_flows
  hidden_size: int = hps.flow_hidden_size
  latent_size: int = hps.latent_size
  info_size: int = hps.latent_size

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

  def make_flow_net(rng):
    _, params = flow_net_init(rng, input_shape=(info_size+latent_size,))
    return params

  def init_fun(rng):
    flow_rng, aux_rng = random.split(rng)

    # Normalizing flow procedure.
    flow_rngs = random.split(flow_rng, 2 * num_flows)
    flow_net1_params = jax.vmap(make_flow_net)(flow_rngs[:num_flows])
    flow_net2_params = jax.vmap(make_flow_net)(flow_rngs[num_flows:])
    norm_flow_params = (flow_net1_params, flow_net2_params)

    # Auxiliary variable procedure.
    qv_rng, rv_rng = random.split(aux_rng)
    _, qv_net_params = model_net_init(qv_rng, input_shape=(info_size+latent_size,))
    _, rv_net_params = model_net_init(rv_rng, input_shape=(info_size+latent_size,))

    params = (norm_flow_params, (qv_net_params, rv_net_params))

    return params

  def sample(rng, z0, x, params):
    """The forward function essentially."""
    norm_flow_params, aux_var_params = params
    qv_net_params, rv_net_params = aux_var_params

    # Auxiliary variable step; forward model q(v|x,z).
    zx = jnp.concatenate((z0, x))
    v0, logqv0 = _forward_aux_var(rng, zx, qv_net_params)

    # Normalizing flow procedure.
    def flow_proc(carry, norm_flow_param):
      """Scan norm flow params, carry over (zT, vT) and compute logdets."""
      #
      #            flow0 -> flow1 -> flow3 ->  ...
      #              \        \         \        \
      #   z0,v0 ->  z1,v1 -> z2,v2 -> z3,v3  ->  ...
      #                \        \         \        \
      #              logdet1   logdet2   logdet3   ...
      #
      last_zT, last_vT = carry
      zT, vT, logdet_flow = _norm_flow(last_zT, last_vT, x, norm_flow_param)
      return (zT, vT), logdet_flow
    
    (zT, vT), logdet_flows = jax.lax.scan(flow_proc, (z0, v0), norm_flow_params)
    logdet_jac = jnp.sum(logdet_flows, axis=0)

    # Auxiliary variable step; reverse model r(v|x,z).
    zx = jnp.concatenate((zT, x))
    logrvT = _reverse_aux_var(zx, vT, rv_net_params)

    # Auxiliary flow correction to log q(z|x).
    logprob = logqv0 - logdet_jac - logrvT

    return zT, logprob
  
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