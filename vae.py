from base64 import encode
from functools import partial
from jax import lax, random, jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.example_libraries import stax

from flow import build_flow, build_aux_flow

from utils import log_bernoulli, log_normal, HyperParams

class VAE:

  def __init__(self, hps: HyperParams):
    self.encoder_init, self.encoder = stax.serial(
      stax.Dense(hps.encoder_width), hps.act_fun,
      stax.Dense(hps.encoder_width), hps.act_fun,
      stax.FanOut(2),
      stax.parallel(
        stax.Dense(hps.latent_size),
        stax.Dense(hps.latent_size),
      ),
    )
    self.decoder_init, self.decoder = stax.serial(
      stax.Dense(hps.decoder_width), hps.act_fun,
      stax.Dense(hps.decoder_width), hps.act_fun,
      stax.Dense(hps.image_size),
    )
    self.init_flow, self.run_flow = build_aux_flow(hps)
    self.hps = hps
  
  def init_params(self, rng):
    encoder_rng, decoder_rng, flow_rng = random.split(rng, num=3)

    encoder_input_shape = (self.hps.image_size,)
    decoder_input_shape = (self.hps.latent_size,)

    _, encoder_params = self.encoder_init(encoder_rng, input_shape=encoder_input_shape)
    _, decoder_params = self.decoder_init(decoder_rng, input_shape=decoder_input_shape)

    if self.hps.has_flow:
      flow_params = self.init_flow(flow_rng)
      params = (encoder_params, decoder_params, flow_params)
    else:
      params = (encoder_params, decoder_params)

    return params
  
  @partial(jit, static_argnums=(0,))
  def run(self, params, x, rng, beta=1.):
    encoder_params = params[0]
    decoder_params = params[1]

    eps_rng, run_flow_rng = random.split(rng, num=2)

    mu, logvar = self.encoder(encoder_params, x)
    eps = random.normal(eps_rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)

    logqz = log_normal(z, mu, logvar)  # log q(z|x)

    # Normalizing flow
    if self.hps.has_flow:
      flow_params = params[2]
      z, logprob = self.run_flow(run_flow_rng, z, flow_params)
      logqz += logprob

    logpz = log_normal(z)  # log p(z)
    # kld = gaussian_kld(mu, logvar)
    kld = logqz - logpz

    logit = self.decoder(decoder_params, z)
    likelihood = log_bernoulli(logit, x)  # log p(x|z)

    elbo = likelihood - beta * kld
    return elbo, logit, likelihood, kld

  @partial(jit, static_argnums=(0,))
  def run_local(self, rng, x, mu, logvar, decoder_params):
    eps = random.normal(rng, mu.shape)
    z = mu + eps * jnp.exp(0.5 * logvar)
    logit = self.decoder(decoder_params, z)

    likelihood = log_bernoulli(logit, x) # log p(x|z)

    logpz = log_normal(z)    # log p(z)
    logqz = log_normal(z, mu, logvar)  # log q(z|x)

    # kld = gaussian_kld(mu, logvar)
    kld = logqz - logpz
    elbo = likelihood - kld
    return elbo, logit, likelihood, kld

  @partial(jit, static_argnums=(0,))
  def run_local_flow(self, rng, x, mu, logvar, flow_params, decoder_params):
    eps_rng, run_flow_rng = random.split(rng, num=2)
    
    eps = random.normal(eps_rng, mu.shape)
    z = mu + eps*jnp.exp(0.5 * logvar)

    logqz = log_normal(z, mu, logvar)  # log q(z|x)
    
    z, logprob = self.run_flow(run_flow_rng, z, flow_params)
    logqz += logprob  # log q(z|x) - log|dzT/dz0| + more correction stuffs.

    logpz = log_normal(z)  # log p(z)
    kld = logqz - logpz

    logit = self.decoder(decoder_params, z)
    likelihood = log_bernoulli(logit, x)  # log p(x|z)

    elbo = likelihood - kld
    return elbo, logit, likelihood, kld

  @partial(jit, static_argnums=(0,))
  def sample(self, params, rng):
    """ Sample from latent space and decode """
    decoder_params = params[1]
    z = random.normal(rng, (self.hps.latent_size,))
    logit = self.decoder(decoder_params, z)
    recon = 1 / (1 + jnp.exp(-logit))
    return recon