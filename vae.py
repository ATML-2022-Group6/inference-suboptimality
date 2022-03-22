from jax import lax, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from flow import build_flow

from utils import log_bernoulli, log_normal, HyperParams

def build_vae(hps: HyperParams):

  encoder_init, encoder = stax.serial(
    stax.Dense(hps.encoder_width), hps.act_fun,
    stax.Dense(hps.encoder_width), hps.act_fun,
    stax.FanOut(2),
    stax.parallel(
      stax.Dense(hps.latent_size),
      stax.Dense(hps.latent_size),
    ),
  )
  decoder_init, decoder = stax.serial(
    stax.Dense(hps.decoder_width), hps.act_fun,
    stax.Dense(hps.decoder_width), hps.act_fun,
    stax.Dense(hps.image_size),
  )
  init_flow, run_flow = build_flow(hps)

  def init_fun(rng, input_shape):
    assert input_shape[-1] == hps.image_size

    encoder_rng, decoder_rng, flow_rng = random.split(rng, num=3)

    _, encoder_params = encoder_init(encoder_rng, input_shape=input_shape)

    decoder_input_shape = input_shape[:-1] + (hps.latent_size,)
    output_shape, decoder_params = decoder_init(decoder_rng, input_shape=decoder_input_shape)

    if hps.has_flow:
      flow_params = init_flow(flow_rng)
      params = (encoder_params, decoder_params, flow_params)
    else:
      params = (encoder_params, decoder_params)

    return params
  
  @jit
  def apply_fun(params, x, rng):
    encoder_params = params[0]
    decoder_params = params[1]

    mu, logvar = encoder(encoder_params, x)
    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)

    logqz = log_normal(z, mu, logvar)  # log q(z|x)    

    # Normalizing flow
    if hps.has_flow:
      flow_params = params[2]
      z, logdetsum = run_flow(z, flow_params)
      logqz -= logdetsum

    logpz = log_normal(z)    # log p(z)
    # kld = gaussian_kld(mu, logvar)
    kld = logqz - logpz

    logit = decoder(decoder_params, z)
    likelihood = log_bernoulli(logit, x) # log p(x|z)
    
    elbo = likelihood - kld # TODO: Warmup const
    return elbo, logit, likelihood, kld
  
  # Sample from latent space and decode
  @jit
  def sample_fun(params, rng):
    decoder_params = params[1]
    z = random.normal(rng, (hps.latent_size,))
    logit = decoder(decoder_params, z)
    recon = 1 / (1 + jnp.exp(-logit))
    return recon
  
  @jit
  def apply_local(rng, x, mu, logvar, decoder_params):

    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    logit = decoder(decoder_params, z)

    likelihood = log_bernoulli(logit, x) # log p(x|z)

    logpz = log_normal(z)    # log p(z)
    logqz = log_normal(z, mu, logvar)  # log q(z|x)    
    
    # kld = gaussian_kld(mu, logvar)
    kld = logqz - logpz
    elbo = likelihood - kld # TODO: Warmup const
    return elbo, logit, likelihood, kld
  
  return init_fun, apply_fun, apply_local, sample_fun