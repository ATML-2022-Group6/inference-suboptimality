from jax import nn, random
from jax import numpy as jnp
from jax.example_libraries import stax

from utils import HyperParams, log_normal

def build_aux_flow(hps: HyperParams):
  """Normalizing flow + auxiliary variable."""
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
        qv_net_params,  # Don't really need now. 
        rv_net_params,
      ),
    )

    return params
  
  def sample(rng, z0, x, params):
    """The forward function essentially."""
    norm_flow_params, aux_var_params = params
    (
      qv_net_params, 
      rv_net_params,
    ) = aux_var_params
    
    zx = jnp.concatenate((z0, x))
    
    # Auxiliary variable, forward model q(v|x,z).
    mean_v0, logvar_v0 = model_net(qv_net_params, zx)
    eps = random.normal(rng, mean_v0.shape)
    v0 = mean_v0 + eps*jnp.exp(0.5 * logvar_v0)
    logqv0 = log_normal(v0, mean_v0, logvar_v0)
    
    # Flow procedure. Currently fixed to 2 flows only.
    zT, vT, logdet_flow1 = _norm_flow(z0, v0, x,
                                      norm_flow_params[0])
    zT, vT, logdet_flow2 = _norm_flow(zT, vT, x,
                                      norm_flow_params[1])
    inverse_logdet_sum = -(logdet_flow1 + logdet_flow2)
    
    # Reverse model: r(v|x,z).
    zx = jnp.concatenate((zT, x))
    logrvT = _reverse_aux_var(zx, vT, rv_net_params)
    
    # Auxiliary flow correction to log q(z|x).
    logprob = logqv0 + inverse_logdet_sum - logrvT
    
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
    μ1, logit_1 = flow_net(net1_params, jnp.concatenate((z, x)))
    σ1 = nn.sigmoid(logit_1)
    v = v*σ1 + μ1 

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    μ2, logit_2 = flow_net(net2_params, jnp.concatenate((v, x)))
    σ2 = nn.sigmoid(logit_2)
    z = z*σ2 + μ2

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
  
  def _reverse_aux_var(zx, v, params):
    """Reverse model in auxiliary variable procedure.
    
    Args:
      `zx` = jnp.concatenate((z, x))
    """
    rv_net_params = params
    
    mean_v, logvar_v = model_net(rv_net_params, zx)
    logrv = log_normal(v, mean_v, logvar_v)
    return logrv
    
  return init_fun, sample