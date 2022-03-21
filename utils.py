from jax import numpy as jnp

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

def log_normal(x, mean, logvar):
  return -0.5 * (jnp.sum(logvar) + jnp.sum((x - mean)**2 / jnp.exp(logvar)))