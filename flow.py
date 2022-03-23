from jax.experimental import loops
from jax import nn, random, lax
from jax import numpy as jnp
from jax.example_libraries import stax

from utils import HyperParams, log_normal

def build_aux_flow(hps: HyperParams):
  """Normalizing flow + auxiliary variable."""
  hidden_size: int = 50
  latent_size: int = hps.latent_size

  # Nets for `_norm_flow()` method; eq. (9) and (10) in Cremer et al.
  #
  # TODO :- Make it for-loop-able, but not sure how to do it the jax way.
  #         Current implementation is fixed for 2 flows only.
  flow1_net1_init, flow1_net1 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size), # ---> μ
      stax.Dense(latent_size), # ---> logit
    ),
  )
  flow1_net2_init, flow1_net2 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h 
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size), # ---> μ
      stax.Dense(latent_size), # ---> logit
    ),
  )
  flow2_net1_init, flow2_net1 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size), # ---> μ
      stax.Dense(latent_size), # ---> logit
    ),
  )
  flow2_net2_init, flow2_net2 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h 
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size), # ---> μ
      stax.Dense(latent_size), # ---> logit
    ),
  )
  
  flow_nets = (
    (flow1_net1, flow1_net2),
    (flow2_net1, flow2_net2),
  )
  
  # Nets for `sample()` method.
  # Forward model: q(v|x, z)
  qv_net_init, qv_net = stax.serial(
    stax.Dense(hidden_size), stax.Elu,
    stax.Dense(hidden_size), stax.Elu,
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_size),
      stax.Dense(latent_size),
    ),
  )
  # Reversible model: r(v|x, z)
  rv_net_init, rv_net = stax.serial(
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
    _, flow1_net1_params = flow1_net1_init(rngs[0], input_shape=(latent_size,))
    _, flow1_net2_params = flow1_net2_init(rngs[1], input_shape=(latent_size,))
    _, flow2_net1_params = flow2_net1_init(rngs[2], input_shape=(latent_size,))
    _, flow2_net2_params = flow2_net2_init(rngs[3], input_shape=(latent_size,))
    
    # Auxiliary variable procedure.
    _, qv_net_params = qv_net_init(rngs[4], input_shape=(latent_size,))
    _, rv_net_params = rv_net_init(rngs[5], input_shape=(latent_size,))

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
  
  def sample(rng, z0, params):
    """The forward function essentially."""
    norm_flow_params, aux_var_params = params
    (
      qv_net_params, 
      rv_net_params,
    ) = aux_var_params
    
    # Auxiliary variable, forward distribution: q(v0)
    # IMPLEMENTATION CONFLICTS :- Cremer initialized mu, logvar with zeros
    # whereas Xuechen get mu, logvar from net transformations on z0.
    #
    # mean_v0, logvar_v0 = qv_net(qv_net_params, z0)  # Xuechen's method
    mean_v0, logvar_v0 = jnp.zeros(latent_size), jnp.zeros(latent_size)  # Cremer's method
    eps = random.normal(rng, mean_v0.shape)
    v0 = mean_v0 + eps*jnp.exp(0.5 * logvar_v0)
    logqv0 = log_normal(v0, mean_v0, logvar_v0)
    
    # Flow procedure. Currently fixed to 2 flows only.
    zT, vT, logdet_flow1 = _norm_flow(z0, v0, 
                                      norm_flow_params[0], 
                                      flow_nets[0])
    zT, vT, logdet_flow2 = _norm_flow(zT, vT,
                                      norm_flow_params[1], 
                                      flow_nets[1])
    inverse_logdet_sum = -(logdet_flow1 + logdet_flow2)
    
    # Reverse distribution: r(vT|x,zT).
    logrvT = _reverse_aux_var(zT, vT, rv_net_params)
    
    # Auxiliary flow correction to log q(z|x).
    logprob = logqv0 + inverse_logdet_sum - logrvT
    
    return zT, logprob
  
  def _norm_flow(z, v, params, nets):
    """
    Real-NVP (Dinh et al.) normalizing flow as defined 
    by equation (9) and (10) in our paper Cremer et al.
    """

    net1_params, net2_params = params
    net1, net2 = nets
    
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
    μ1, logit_1 = net1(net1_params, z)
    σ1 = nn.sigmoid(logit_1)
    v = v*σ1 + μ1 

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    μ2, logit_2 = net2(net2_params, v)
    σ2 = nn.sigmoid(logit_2)
    z = z*σ2 + μ2 

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
  
  def _reverse_aux_var(zT, vT, params):
    """Reverse model in auxiliary variable procedure."""
    rv_net_params = params
    
    mean_vT, logvar_vT = rv_net(rv_net_params, zT)
    logrvT = log_normal(vT, mean_vT, logvar_vT)
    return logrvT
    
  return init_fun, sample


def build_flow(hps: HyperParams):
  """Real-NVP (Dinh et al.) without auxiliary variable."""
  hidden_size: int = 50
  latent_size: int = hps.latent_size
  
  # With notation as in Dinh et al. (Real NVP paper),
  # `latent_split_size` = D//2 where the latent variable z is 
  # indexed as 1:d, d+1:D.
  latent_split_size: int = latent_size // 2

  # Nets for `_norm_flow()` method; eq. (9) and (10) in Cremer et al.
  #
  # TODO :- Make it for-loop-able, but not sure how to do it the jax way.
  #         Current implementation is fixed for 2 flows only.
  flow1_net1_init, flow1_net1 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_split_size), # ---> μ
      stax.Dense(latent_split_size), # ---> logit
    ),
  )
  flow1_net2_init, flow1_net2 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h 
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_split_size), # ---> μ
      stax.Dense(latent_split_size), # ---> logit
    ),
  )
  flow2_net1_init, flow2_net1 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_split_size), # ---> μ
      stax.Dense(latent_split_size), # ---> logit
    ),
  )
  flow2_net2_init, flow2_net2 = stax.serial(
    stax.Dense(hidden_size), stax.Elu, # --> h 
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(latent_split_size), # ---> μ
      stax.Dense(latent_split_size), # ---> logit
    ),
  )
  
  flow_nets = (
    (flow1_net1, flow1_net2),
    (flow2_net1, flow2_net2),
  )
  
  def init_fun(rng):
    rngs = random.split(rng, num=6)

    # Flow procedure.
    _, flow1_net1_params = flow1_net1_init(rngs[0], input_shape=(latent_split_size,))
    _, flow1_net2_params = flow1_net2_init(rngs[1], input_shape=(latent_split_size,))
    _, flow2_net1_params = flow2_net1_init(rngs[2], input_shape=(latent_split_size,))
    _, flow2_net2_params = flow2_net2_init(rngs[3], input_shape=(latent_split_size,))
    
    params = (
      (
        (flow1_net1_params, flow1_net2_params),
        (flow2_net1_params, flow2_net2_params),
      ),
      None,  # To make API consistent with auxiliary flow.
    )

    return params
  
  def sample(rng, z0, params):
    """The forward function essentially.
    
    `rng`: Not needed, only for API consistency with auxiliary flow.
    """
    norm_flow_params, _ = params
    
    # With notation as in Dinh et al. (Real NVP paper),
    # split z with index 1:d, d+1:D split.
    z1, z2 = z0[:latent_split_size], z0[latent_split_size:]
  
    # Flow procedure. Currently fixed to 2 flows only.
    z1, z2, logdet_flow1 = _norm_flow(z1, z2, 
                                      norm_flow_params[0], 
                                      flow_nets[0])
    z1, z2, logdet_flow2 = _norm_flow(z1, z2, 
                                      norm_flow_params[1], 
                                      flow_nets[1])
    inverse_logdet_sum = -(logdet_flow1 + logdet_flow2)
    
    # Concatenate the 1:d, d+1:D z-split.
    z = jnp.concatenate([z1, z2], axis=0)
    
    # Flow correction to log q(z|x).
    logprob = inverse_logdet_sum  # This assignment is for API consistency with aux flow.
    
    return z, logprob
  
  def _norm_flow(z, v, params, nets):
    """
    Real-NVP (Dinh et al.) normalizing flow as defined 
    by equation (9) and (10) in our paper Cremer et al.
    """
    net1_params, net2_params = params
    net1, net2 = nets
    
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
    μ1, logit_1 = net1(net1_params, z)
    σ1 = nn.sigmoid(logit_1)
    v = v*σ1 + μ1 

    # Equation (10) in paper. No difference doing z_1 or z_2 first in the coupling.
    μ2, logit_2 = net2(net2_params, v)
    σ2 = nn.sigmoid(logit_2)
    z = z*σ2 + μ2 

    logdet_v = jnp.sum(logit_1 - nn.softplus(logit_1))
    logdet_z = jnp.sum(logit_2 - nn.softplus(logit_2))
    logdet = logdet_v + logdet_z

    return z, v, logdet
    
  return init_fun, sample