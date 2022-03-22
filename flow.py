from dataclasses import dataclass
from jax import nn, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from utils import HyperParams, log_normal

def build_aux_flow(hps: HyperParams):
  """Normalizing flow + auxiliary variable."""
  n_flows: int = 1
  hidden_size: int = 50
  latent_size: int = hps.latent_size
  
  # With notation as in Dinh et al. (Real NVP paper),
  # `latent_split_size` = D//2 where the latent variable z is 
  # indexed as 1:d, d+1:D.
  latent_split_size: int = latent_size // 2

  # Nets for `_norm_flow()` method; eq. (9) and (10) in Cremer et al.
  h1_net_init, h1_net = stax.serial(stax.Dense(hidden_size), stax.Elu)
  h2_net_init, h2_net = stax.serial(stax.Dense(hidden_size), stax.Elu)
  μ1_net_init, μ1_net = stax.Dense(latent_split_size)
  μ2_net_init, μ2_net = stax.Dense(latent_split_size)
  σ1_net_init, σ1_net = stax.Dense(latent_split_size)
  σ2_net_init, σ2_net = stax.Dense(latent_split_size)
  
  # Nets for `sample()` method.
  rv_net_init, rv_net = stax.serial(
    stax.Dense(hidden_size), stax.Elu,
    stax.Dense(hidden_size), stax.Elu,
    stax.Dense(2*latent_size),  # Twice the size so we can split into mean & logvar resp.
  )
  
  def init_fun(rng):
    rngs = random.split(rng, num=7)

    h1_size, h1_net_params = h1_net_init(rngs[0], input_shape=(latent_split_size,))
    _, μ1_net_params = μ1_net_init(rngs[1], input_shape=h1_size)
    _, σ1_net_params = σ1_net_init(rngs[2], input_shape=h1_size)

    h2_size, h2_net_params = h2_net_init(rngs[3], input_shape=(latent_split_size,))
    _, μ2_net_params = μ2_net_init(rngs[4], input_shape=h2_size)
    _, σ2_net_params = σ2_net_init(rngs[5], input_shape=h2_size)

    _, rv_net_params = rv_net_init(6, input_shape=(2*latent_size,))

    params = (
      (
        h1_net_params, 
        μ1_net_params, 
        σ1_net_params,
        h2_net_params, 
        μ2_net_params, 
        σ2_net_params,
      ),
      rv_net_params,
    )

    return params
  
  def sample(rng, z0, params):
    """The forward function essentially."""
    norm_flow_params, aux_var_params = params
    
    # Auxiliary variable: q(v0)
    # IMPLEMENTATION COMMENTS :- Cremer initialized mu, logvar with zeros
    # whereas Xuechen get mu, logvar from net transformations on z0.
    mu, logvar = jnp.zeros(latent_size), jnp.zeros(latent_size)
    eps = random.normal(rng, mu.shape)
    v0 = mu + eps*jnp.exp(0.5 * logvar)
    logqv0 = log_normal(v0, mu, logvar)
    
    # Flow procedure
    logdetsum = 0.
    # TODO :- For loop on the number of flows (has to have as many 
    # neural nets). For this implementation may need to bring current net inits
    # `_norm_flow()` to parent function init.
    zT, vT, logdet = _norm_flow(norm_flow_params, z0, v0)
    logdetsum += logdet
    
    # Reverse distribution: r(vT|x,zT)
    logrvT = _aux_var(aux_var_params, zT, vT)
    
    # Auxiliary flow correction to log q(z|x).
    logprob = logqv0 - logdetsum - logrvT
    return zT, logprob
  
  def _norm_flow(params, z, v):
    """
    Real-NVP (Dinh et al.) normalizing flow as defined 
    by equation (9) and (10) in our paper Cremer et al.
    """
    (
      h1_net_params, 
      μ1_net_params, 
      σ1_net_params,
      h2_net_params, 
      μ2_net_params, 
      σ2_net_params,
    ) = params
    
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
    h1 = h1_net(h1_net_params, z)
    μ1 = μ1_net(μ1_net_params, h1)
    logit_1 = σ1_net(σ1_net_params, h1)
    v = v*nn.sigmoid(logit_1) + μ1  # TODO :- Add nn.elu to μ1 and multiply with `grad_fn(z)`.

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    h2 = h2_net(h2_net_params, v)
    μ2 = μ2_net(μ2_net_params, h2)
    logit_2 = σ2_net(σ2_net_params, h2)
    z = z*nn.sigmoid(logit_2) + μ2  # TODO :- Add nn.elu to μ2 and multiply with `grad_fn(z)`.

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
  
  def _aux_var(params, zT, vT):
    """Auxiliary variable procedure."""
    rv_net_params = params
    
    out = rv_net(rv_net_params, zT)
    mean_vT, logvar_vT = out[:latent_size], out[latent_size:]
    logrvT = log_normal(vT, mean_vT, logvar_vT)
    
    return logrvT
    
  return init_fun, sample




def build_flow(hps: HyperParams):
  """Real-NVP (Dinh et al.) without auxiliary variable."""
  n_flows: int = 1
  hidden_size: int = 50
  latent_size: int = hps.latent_size
  
  # With notation as in Dinh et al. (Real NVP paper),
  # `latent_split_size` = D//2 where the latent variable z is 
  # indexed as 1:d, d+1:D.
  latent_split_size: int = latent_size // 2

  h1_net_init, h1_net = stax.serial(stax.Dense(hidden_size), stax.Elu)
  h2_net_init, h2_net = stax.serial(stax.Dense(hidden_size), stax.Elu)
  μ1_net_init, μ1_net = stax.Dense(latent_split_size)
  μ2_net_init, μ2_net = stax.Dense(latent_split_size)
  σ1_net_init, σ1_net = stax.Dense(latent_split_size)
  σ2_net_init, σ2_net = stax.Dense(latent_split_size)
  
  def init_fun(rng):
    rngs = random.split(rng, num=6)

    h1_size, h1_net_params = h1_net_init(rngs[0], input_shape=(latent_split_size,))
    μ1_size, μ1_net_params = μ1_net_init(rngs[1], input_shape=h1_size)
    σ1_size, σ1_net_params = σ1_net_init(rngs[2], input_shape=h1_size)

    h2_size, h2_net_params = h2_net_init(rngs[3], input_shape=(latent_split_size,))
    μ2_size, μ2_net_params = μ2_net_init(rngs[4], input_shape=h2_size)
    σ2_size, σ2_net_params = σ2_net_init(rngs[5], input_shape=h2_size)

    params = (
      h1_net_params, μ1_net_params, σ1_net_params,
      h2_net_params, μ2_net_params, σ2_net_params
    )
    
    params = (
      h1_net_params, 
      μ1_net_params, 
      σ1_net_params,
      h2_net_params, 
      μ2_net_params, 
      σ2_net_params,
    )

    return params
  
  def sample(z0, params):
    """The forward function essentially."""
    
    # With notation as in Dinh et al. (Real NVP paper),
    # split z with index 1:d, d+1:D split.
    z1, z2 = z0[:latent_split_size], z0[latent_split_size:]
    
    logdetsum = 0.

    # TODO :- For loop on the number of flows (has to have as many 
    # neural nets). For this implementation may need to bring current net inits
    # `_norm_flow()` to parent function init.
    z1, z2, logdet = _norm_flow(params, z1, z2)
    logdetsum += logdet
    
    # Concatenate the 1:d, d+1:D z-split (will need for aux flow!).
    z = jnp.concatenate([z1, z2], axis=0)
    
    # Flow correction to log q(z|x).
    logprob = -logdetsum
    
    return z, logprob
  
  def _norm_flow(params, z, v):
    """
    Real-NVP (Dinh et al.) normalizing flow as defined 
    by equation (9) and (10) in our paper Cremer et al.
    """
    (
      h1_net_params, 
      μ1_net_params, 
      σ1_net_params,
      h2_net_params, 
      μ2_net_params, 
      σ2_net_params,
    ) = params
    
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
    h1 = h1_net(h1_net_params, z)
    μ1 = μ1_net(μ1_net_params, h1)
    logit_1 = σ1_net(σ1_net_params, h1)
    v = v*nn.sigmoid(logit_1) + μ1  # TODO :- Add nn.elu to μ1 and multiply with `grad_fn(z)`.

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    h2 = h2_net(h2_net_params, v)
    μ2 = μ2_net(μ2_net_params, h2)
    logit_2 = σ2_net(σ2_net_params, h2)
    z = z*nn.sigmoid(logit_2) + μ2  # TODO :- Add nn.elu to μ2 and multiply with `grad_fn(z)`.

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
    
  return init_fun, sample