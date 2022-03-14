from jax import lax, random
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from dataclasses import dataclass

@dataclass
class HyperParams:
  latent_size = 50
  image_size = 28 * 28
  encoder_width = 200
  decoder_width = 200
  act_fun = stax.Elu

def log_bernoulli(logit, target):
  loss = -jnp.max(logit, 0) + jnp.multiply(logit, target) - jnp.log1p(jnp.exp(-jnp.abs(logit)))
  return jnp.sum(loss)

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

  def init_fun(rng: random.KeyArray, input_shape: tuple[int,...]):
    batch_shape = input_shape[:-1]
    assert input_shape[-1] == hps.image_size

    encoder_rng, decoder_rng = random.split(rng)
    _, encoder_params = encoder_init(encoder_rng, input_shape=input_shape)

    decoder_input_shape = input_shape[:-1] + (hps.latent_size,)
    output_shape, decoder_params = decoder_init(decoder_rng, input_shape=decoder_input_shape)

    params = (encoder_params, decoder_params)
    return output_shape, params
  
  def apply_fun(params: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, **kwargs):
    rng = kwargs["rng"]
    encoder_params, decoder_params = params

    mu, logvar = encoder(encoder_params, x)
    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    logit = decoder(decoder_params, z)

    logpx = log_bernoulli(logit, x) # log p(x|z)
    logpz = stats.norm.logpdf(z)    # log p(z)
    logqz = stats.norm.logpdf(eps)  # log q(z|x)

    elbo = logpx + logpz - logqz # TODO: Warmup const

    return jnp.mean(elbo)
  
  return init_fun, apply_fun

