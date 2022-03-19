from jax import lax, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax
from jax.tree_util import tree_map

from dataclasses import dataclass

from flow import build_flow
import pickle

@dataclass
class HyperParams:
  latent_size: int = 50
  image_size: int = 28 * 28
  encoder_width: int = 200
  decoder_width: int = 200
  act_fun: tuple = stax.Elu
  has_flow: bool = False

def log_bernoulli(logit, target):
  # Numerically equivalent to Binary Cross Entropy loss (TODO: How?)
  bce_loss = jnp.maximum(logit, 0.) - logit * target + jnp.logaddexp(0., -jnp.abs(logit))
  return -jnp.sum(bce_loss)

# Equivalent to below
# def log_sigmoid(x):
#   return -jnp.logaddexp(0, -x)
# def bce(logit, target):
#   return jnp.sum( target * log_sigmoid(logit) + (1 - target) * log_sigmoid(-logit) )

def gaussian_kld(mu, logvar):
  return -0.5 * jnp.sum(1. + logvar - mu**2. - jnp.exp(logvar))

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
  
  @jit
  def apply_fun(params, x, rng):
    encoder_params, decoder_params = params

    mu, logvar = encoder(encoder_params, x)
    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    logit = decoder(decoder_params, z)

    likelihood = log_bernoulli(logit, x) # log p(x|z)

    logpz = jnp.sum(stats.norm.logpdf(z))    # log p(z)
    logqz = jnp.sum(stats.norm.logpdf(eps))  # log q(z|x)    
    
    # Normalizing flow
    if hps.has_flow:
      sample_flow = build_flow(hps)
      logqz += sample_flow(rng, mu, logvar, k=1)
    
    # kld = gaussian_kld(mu, logvar)
    kld = logqz - logpz
    elbo = likelihood - kld # TODO: Warmup const
    return elbo, logit, likelihood, kld
  
  # Sample from latent space and decode
  @jit
  def sample_fun(params, rng):
    _, decoder_params = params
    z = random.normal(rng, (hps.latent_size,))
    logit = decoder(decoder_params, z)
    recon = 1 / (1 + jnp.exp(-logit))
    return recon
  
  return init_fun, apply_fun, sample_fun

def save_params(file_name, params):
  with open(file_name, "wb") as f:
    pickle.dump(params, f)

def load_params(file_name):
  with open(file_name, "rb") as f:
    params = pickle.load(f)
    # convert NP arrays to Jax arrays
    return tree_map(lambda param: jnp.array(param), params)