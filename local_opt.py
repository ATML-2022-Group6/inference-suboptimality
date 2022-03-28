from mimetypes import init
import numpy as np

import jax
from jax.scipy.special import logsumexp
from jax import numpy as jnp
from jax.example_libraries import optimizers
from jax import jit
from jax import random

from functools import partial

from tqdm.notebook import tqdm, trange

from vae import VAE

@partial(jit, static_argnums=(0,1))
def batch_iwae(model: VAE, num_samples, images, rng, enc_paramss, dec_params):
  
  def run_iwae(rng, image, enc_params):
    rngs = random.split(rng, num_samples)
    iw_log_summand, _, _, _ = jax.vmap(model.run_local,
        in_axes=(0, None, None, None)
      )( rngs, image, enc_params, dec_params )

    iwelbo = logsumexp(iw_log_summand) - jnp.log(num_samples)
    return iwelbo

  rngs = random.split(rng, images.shape[0])
  iwaes = jax.vmap(run_iwae, in_axes=(0, 0, 0))(rngs, images, enc_paramss)

  return jnp.mean(iwaes)

@partial(jit, static_argnums=(0,1))
def run_epoch(
  model: VAE, optimizer: optimizers.Optimizer,
  epoch, rng, opt_state, decoder_params, batch
):

  def batch_loss_fn(enc_params):
    rngs = random.split(rng, batch.shape[0])
    elbos, _, _, _ = jax.vmap(model.run_local, in_axes=(0, 0, 0, None))(rngs, batch, enc_params, decoder_params)
    return -jnp.mean(elbos)

  enc_params = optimizer.params_fn(opt_state)
  loss, g = jax.value_and_grad(batch_loss_fn)(enc_params)

  return optimizer.update_fn(epoch, g, opt_state), loss

def optimize_local_batch(
  model: VAE,
  trained_params,
  batch,
  optimizer: optimizers.Optimizer,
  debug = False,
  num_samples = 100, # for IWAE
  # Early stopping parameters
  check_every = 100,
  patience = 10,
  epsilon = 0.05,
):
  encoder_params = trained_params[0]
  decoder_params = trained_params[1]

  # use trained encoder to initialise mu0, logvar0
  mu0, logvar0 = jax.vmap(model.encoder, in_axes=(None, 0))(encoder_params, batch)

  # whether to optimise local flow or local FFG
  if model.hps.has_flow:

    # take trained flow params if available, otherwise sample randomly
    if len(trained_params) > 2:
      flow_params0 = trained_params[2]
    else:
      flow_rng = random.PRNGKey(0)
      batch_size = len(batch)
      flow_params0 = jax.vmap(model.init_flow)(random.split(flow_rng, batch_size))
    init_params = (mu0, logvar0, flow_params0)

  else:
    init_params = (mu0, logvar0)

  opt_state = optimizer.init_fn(init_params)

  best_avg, sentinel = 1e20, 0
  train_loss = []

  with trange(1,10**6) as t:
    for epoch in t:
      epoch_rng = random.PRNGKey(epoch)
      opt_state, loss = run_epoch(model, optimizer, epoch-1, epoch_rng, opt_state, decoder_params, batch)
      train_loss.append(loss)

      if epoch % check_every == 0:
        last_avg = jnp.mean(jnp.array(train_loss))
        t.set_postfix(avg_loss=-last_avg)
        if debug:
          print("Epoch {:.4f} - ELBO {:.4f}".format(epoch, -last_avg))
        if last_avg < best_avg - epsilon:
          sentinel, best_avg = 0, last_avg
        else:
          sentinel += 1
          if sentinel >= patience: break
        train_loss = []

  final_params = optimizer.params_fn(opt_state)
  final_elbo = -train_loss[-1]
  iwae_rng = random.PRNGKey(0)
  final_iwae = batch_iwae(model, num_samples, batch, iwae_rng, final_params, decoder_params)

  return final_elbo, final_iwae

def local_opt(
  model: VAE, dataset, trained_params,
  optimizer: optimizers.Optimizer = optimizers.adam(step_size=1e-3, eps=1e-4),
):
  elbo_record, iwae_record = [], []
  print("Optimising Local", "Flow" if model.hps.has_flow else "FFG", "...")
  for i, batch in enumerate(tqdm(dataset)):
    elbo, iwae = optimize_local_batch(model, trained_params, batch, optimizer)
    elbo_record.append(elbo)
    iwae_record.append(iwae)
    print("Batch {}, ELBO {:.4f}, IWAE {:.4f}".format(i+1, elbo, iwae))
  print("Average ELBO {:.4f}".format(np.nanmean(elbo_record)))
  print("Average IWAE {:.4f}".format(np.nanmean(iwae_record)))