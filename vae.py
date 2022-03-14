from jax import lax, random
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from dataclasses import dataclass

@dataclass
class HyperParams:
  latent_size: int = 50
  image_size: int = 28 * 28
  encoder_width: int = 200
  decoder_width: int = 200
  act_fun: tuple = stax.Elu

def log_bernoulli(logit, target):
  # loss = -jnp.max(logit, 0) + jnp.multiply(logit, target) - jnp.log(1. + jnp.exp(-jnp.abs(logit)))
  # loss = -jnp.max(logit, 0) + jnp.multiply(logit, target) - jnp.logaddexp(0, -jnp.abs(logit))
  # return jnp.sum(loss)
  return -jnp.sum(jnp.logaddexp(0., (1 - target * 2) * logit))

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

  def init_fun(rng, input_shape):
    assert input_shape[-1] == hps.image_size

    encoder_rng, decoder_rng = random.split(rng)
    _, encoder_params = encoder_init(encoder_rng, input_shape=input_shape)

    decoder_input_shape = input_shape[:-1] + (hps.latent_size,)
    output_shape, decoder_params = decoder_init(decoder_rng, input_shape=decoder_input_shape)

    params = (encoder_params, decoder_params)
    return params
  
  def apply_fun(params, x, rng):
    encoder_params, decoder_params = params

    mu, logvar = encoder(encoder_params, x)
    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    logit = decoder(decoder_params, z)

    logpx = log_bernoulli(logit, x) # log p(x|z)

    logpz = jnp.sum(stats.norm.logpdf(z))    # log p(z)
    logqz = jnp.sum(stats.norm.logpdf(eps))  # log q(z|x)    
    kld = logqz - logpz

    elbo = logpx - kld # TODO: Warmup const

    return elbo, logit, logpx, logpz, logqz
  
  # Sample from latent space and decode
  def sample_fun(params, rng):
    _, decoder_params = params
    z = random.normal(rng, (hps.latent_size,))
    logit = decoder(decoder_params, z)
    recon = 1 / (1 + jnp.exp(-logit))
    return recon
  
  return init_fun, apply_fun, sample_fun

