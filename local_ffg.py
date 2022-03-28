from sklearn.metrics import pair_confusion_matrix

import numpy as np

import jax

from jax.example_libraries import stax
from jax.scipy.special import logsumexp
from jax.scipy import stats
from jax import numpy as jnp
from jax.example_libraries import optimizers
from tqdm.notebook import tqdm, trange
from jax import jit
from jax import random

from functools import partial

import time

from vae import VAE

# TODO: Convert this code
#
# def iwelbo_fn(rng, enc_params, decoder_params, image):
#     rngs = random.split(rng, num_samples)
#     mu, logvar = enc_params
#     iw_log_summand, _, _, _ = jax.vmap(run_vae_local, in_axes=(0, None, None, None, None))( rngs, image, mu, logvar, decoder_params )

#     K = num_samples
#     iwelbo_K = logsumexp(iw_log_summand) - jnp.log(K)
#     return iwelbo_K

# def batch_iwelbo_fn(rng, enc_params, decoder_params, images):
#   rngs = random.split(rng, batch_size)
#   return jnp.mean(jax.vmap(iwelbo_fn, in_axes=(0, 0, None, 0))(rngs, enc_params, decoder_params, images))

# @jit
# def run_iwelbo_epoch(epoch, rng,opt_state, decoder_params, batch):

#   def body_fn(opt_state, args):
#     idx, rng, batch = args
#     enc_params = get_params(opt_state)
#     loss = batch_iwelbo_fn(rng, enc_params,decoder_params, batch)
#     return loss

#   scan_args = (epoch, rng, batch)
#   losses = body_fn(opt_state, scan_args)

#   return losses

@partial(jit, static_argnums=(0,1))
def run_epoch(
  model: VAE, optimizer: optimizers.Optimizer,
  epoch, rng, opt_state, decoder_params, batch
):

  def loss_fn(rng, enc_params, decoder_params, image):
    mu, logvar = enc_params
    elbo, _, _, _= model.run_local(rng, image, mu, logvar, decoder_params)
    return -elbo

  def batch_loss_fn(rng, enc_params, decoder_params, images):
    rngs = random.split(rng, images.shape[0])
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, None, 0))(rngs, enc_params, decoder_params, images))

  enc_params = optimizer.params_fn(opt_state)
  loss, g = jax.value_and_grad(batch_loss_fn, argnums=1)(rng, enc_params, decoder_params, batch)

  return optimizer.update_fn(epoch, g, opt_state), loss

def optimize_local_gaussian(
  model: VAE,
  decoder_params,
  batch,
  optimizer: optimizers.Optimizer = optimizers.adam(step_size=1e-3, eps=1e-4),
  check_every=100,
  sentinel_thres=10,
  debug=False
):
  batch_size = len(batch)
  latent_size = model.hps.latent_size

  mu0 = jnp.zeros((batch_size,latent_size))
  logvar0 = jnp.zeros((batch_size, latent_size))
  init_params = (mu0, logvar0)

  opt_state = optimizer.init_fn(init_params)

  best_avg, sentinel = 999999, 0
  time_ = time.time()
  train_loss, plot_elbo = [], []

  for epoch in trange(1, 999999):
    epoch_rng = random.PRNGKey(epoch)
    opt_state, loss = run_epoch(model, optimizer, epoch-1, epoch_rng, opt_state, decoder_params, batch)
    train_loss.append(loss)

    if epoch % check_every == 0:
      last_avg = jnp.mean(jnp.array(train_loss))
      if debug:
        print(
          'Epoch %d, time elapse %.4f, last avg %.4f, prev best %.4f\n' % \
          (epoch, time.time()-time_, -last_avg, -best_avg)
        )
      if last_avg < best_avg:
        sentinel, best_avg = 0, last_avg
      else:
          sentinel += 1
      if sentinel > sentinel_thres:
          break
      train_loss = []
      plot_elbo.append(-last_avg)
      time_ = time.time()

  final_elbo = -train_loss[-1]
  return final_elbo, plot_elbo

def local_ffg(
  model: VAE,
  dataset,
  trained_params,
):
  elbo_record = []
  time_ = time.time()
  decoder_params = trained_params[1]

  for i, batch in enumerate(tqdm(dataset)):
      elbo, _ = optimize_local_gaussian(model, decoder_params, batch)
      elbo_record.append(elbo)
      print('Local opt w/ ffg, batch %d, time elapse %.4f, ELBO %.4f' % (i+1, time.time()-time_, elbo))
      print('mean of ELBO so far %.4f' % np.nanmean(elbo_record))
      time_ = time.time()

  print('Finishing...')
  print('Average ELBO %.4f' % np.nanmean(elbo_record))

  return elbo_record