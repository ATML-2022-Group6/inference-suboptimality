from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.example_libraries import optimizers
from jax.scipy.special import logsumexp
from tqdm.notebook import tqdm, trange

from vae import VAE

@dataclass
class LocalHyperParams:
  # samples and LR per iteration
  learning_rate: float = 1e-3
  mc_samples: int = 100

  display_epoch: int = 100
  debug: bool = True

  # IWAE evaluation
  iwae_samples: int = 100

  # Early stopping
  max_epochs: int = 100000
  patience: int = 10
  es_epsilon: float = 0.05

@partial(jit, static_argnums=(0,1))
def batch_iwae(model: VAE, num_samples, images, rng, enc_paramss, dec_params):

  if model.hps.has_flow:
    enc_axes = (0, 0, None)
  else:
    enc_axes = (0, 0)

  def run_iwae(rng, image, enc_params):
    rngs = random.split(rng, num_samples)
    elbos, _, _, _ = jax.vmap(model.run_local,
        in_axes=(0, None, None, None)
      )( rngs, image, enc_params, dec_params )

    iwelbo = logsumexp(elbos) - jnp.log(num_samples)
    elbo = jnp.mean(elbos)

    return iwelbo, elbo

  rngs = random.split(rng, images.shape[0])
  iwaes, elbos = jax.vmap(run_iwae, in_axes=(0, 0, enc_axes))(rngs, images, enc_paramss)

  return jnp.mean(iwaes), jnp.mean(elbos)

@partial(jit, static_argnums=(0,1,2))
def run_epoch(
  model: VAE, optimizer: optimizers.Optimizer, num_samples,
  epoch, rng, opt_state, decoder_params, batch
):

  if model.hps.has_flow:
    enc_axes = (0, 0, None)
  else:
    enc_axes = (0, 0)

  def loss_fn(rng, image, enc_params):
    rngs = random.split(rng, num_samples)
    elbos, _, _, _ = jax.vmap(model.run_local, in_axes=(0, None, None, None))(rngs, image, enc_params, decoder_params)
    return -jnp.mean(elbos)

  def batch_loss_fn(enc_paramss):
    rngs = random.split(rng, batch.shape[0])
    losses = jax.vmap(loss_fn, in_axes=(0, 0, enc_axes))(rngs, batch, enc_paramss)
    return jnp.mean(losses)

  enc_paramss = optimizer.params_fn(opt_state)
  loss, g = jax.value_and_grad(batch_loss_fn)(enc_paramss)

  return optimizer.update_fn(epoch, g, opt_state), loss

def optimize_local_batch(hps: LocalHyperParams, model: VAE, trained_params, batch):
  encoder_params = trained_params[0]
  decoder_params = trained_params[1]
  batch_size = len(batch)
  latent_size = model.hps.latent_size

  # use trained encoder to initialise mu0, logvar0
  enc_out = jax.vmap(model.encoder, in_axes=(None, 0))(encoder_params, batch)
  # enc_out could be a triple (w/ x_info as third part)
  mu0 = enc_out[0]
  logvar0 = enc_out[1]

  # or use prior to initialise means
  # mu0 = logvar0 = jnp.zeros((batch_size, latent_size))

  # whether to optimise local flow or local FFG
  if model.hps.has_flow:

    # Note: We use a *single* flow network across the batch
    flow_rng = random.PRNGKey(0)
    flow_params0 = model.init_flow(flow_rng)
    init_params = (mu0, logvar0, flow_params0)

  else:
    init_params = (mu0, logvar0)

  optimizer = optimizers.adam(step_size=hps.learning_rate, eps=1e-4)
  opt_state = optimizer.init_fn(init_params)

  best_avg, sentinel = 1e20, 0
  train_loss = []

  with trange(1, hps.max_epochs+1) as t:
    for epoch in t:
      epoch_rng = random.PRNGKey(epoch)
      opt_state, loss = run_epoch(
        model, optimizer, hps.mc_samples,
        epoch-1, epoch_rng, opt_state, decoder_params, batch
      )
      train_loss.append(loss)

      if epoch % hps.display_epoch == 0:
        last_avg = jnp.mean(jnp.array(train_loss))
        t.set_postfix(avg_loss=-last_avg)
        if hps.debug:
          print("Epoch {:.4f} - ELBO {:.4f}".format(epoch, -last_avg))
        if last_avg < best_avg - hps.es_epsilon:
          sentinel, best_avg = 0, last_avg
        else:
          sentinel += 1
          if sentinel >= hps.patience: break
        train_loss = []

  final_params = optimizer.params_fn(opt_state)
  iwae_rng = random.PRNGKey(0)
  final_iwae, final_elbo = batch_iwae(model, hps.iwae_samples, batch, iwae_rng, final_params, decoder_params)

  return final_elbo, final_iwae, final_params

def local_opt(hps: LocalHyperParams, model: VAE, dataset, trained_params):
  elbo_record, iwae_record, param_record = [], [], []
  print("Optimising Local", "Flow" if model.hps.has_flow else "FFG", "...")
  for i, batch in enumerate(tqdm(dataset)):
    elbo, iwae, params = optimize_local_batch(hps, model, trained_params, batch)
    elbo_record.append(elbo)
    iwae_record.append(iwae)
    param_record.append(params)
    print("Batch {}, ELBO {:.4f}, IWAE {:.4f}".format(i+1, elbo, iwae))
  print("Average ELBO {:.4f}".format(np.nanmean(elbo_record)))
  print("Average IWAE {:.4f}".format(np.nanmean(iwae_record)))
  return elbo_record, iwae_record, param_record
