from jax import numpy as jnp
from jax.example_libraries import stax

from dataclasses import dataclass

from jax.tree_util import tree_map
import pickle

@dataclass
class HyperParams:
  image_size: int = 28 * 28
  latent_size: int = 50
  encoder_hidden: tuple = (200, 200)
  decoder_hidden: tuple = (200, 200)
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

def log_normal(x, mean=0., logvar=0.):
  """ Gaussian log-pdf (correct only up to constant) """
  return -0.5 * jnp.sum(logvar + (x - mean)**2 / jnp.exp(logvar))

def save_params(file_name, params):
  with open(file_name, "wb") as f:
    pickle.dump(params, f)

def load_params(file_name):
  with open(file_name, "rb") as f:
    params = pickle.load(f)
    # convert NP arrays to Jax arrays
    return tree_map(lambda param: jnp.array(param), params)