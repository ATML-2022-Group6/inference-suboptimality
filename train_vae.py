from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.example_libraries import optimizers
from tqdm.notebook import tqdm, trange

import utils
from vae import VAE

@dataclass
class TrainHyperParams:
  num_epochs: int = 5000
  display_epoch: int = 10

  # num ELBO samples for final evaluation
  eval_elbos: int = 1000

  # always save final params to save_dir/params_{time}.pkl
  # optionally save params every `save_epoch` epochs if positive
  save_epoch: int = -1
  save_dir: str = "."

  init_seed: int = 0

  kl_annealing: bool = False
  kl_threshold: int = 500

  # schedule in terms of epoch, default to:
  # exponential decay 1e-3 to 1e-4 over 3000 epochs
  lr_schedule: Callable = lambda epoch: jnp.maximum(1e-3 * 0.1 ** (epoch / 3000), 1e-4)

  early_stopping: bool = False
  patience: int = 10
  es_epsilon: float = 0.05

@partial(jit, static_argnums=(0,))
def dataset_elbo(model, dataset, rng, params):
  def batch_elbo(images, rng):
    rngs = random.split(rng, images.shape[0])
    elbos, _, _, _ = jax.vmap(model.run, in_axes=(None, 0, 0))(params, images, rngs)
    return jnp.mean(elbos)
  
  rngs = random.split(rng, dataset.shape[0])
  elbos = jax.vmap(batch_elbo)(dataset, rngs)
  return jnp.mean(elbos)

def elbo_estimate(model, num_samples, dataset, rng, params):
  """ Compute mean and estimated std. dev. (of mean) of computing the dataset ELBO `num_sample` times. """
  
  rngs = random.split(rng, num_samples)
  elbos = []
  for i in trange(num_samples):
    elbo = dataset_elbo(model, dataset, rngs[i], params)
    elbos.append(elbo)
  elbos = jnp.array(elbos)

  elbo_var = jnp.var(elbos)
  mean_stddev = (elbo_var / num_samples) ** 0.5

  return jnp.mean(elbos), mean_stddev

def train_vae(hps: TrainHyperParams, model: VAE, train_batches, test_batches):
  
  num_train_batches = train_batches.shape[0]
  batch_size = train_batches.shape[1]

  kl_annealing = hps.kl_annealing
  kl_threshold = hps.kl_threshold
  save_epoch = hps.save_epoch
  save_dir = hps.save_dir
  early_stopping = hps.early_stopping
  es_epsilon = hps.es_epsilon
  patience = hps.patience
  num_epochs = hps.num_epochs
  display_epoch = hps.display_epoch
  
  # lr_schedule in terms of batch index
  lr_schedule = lambda i: hps.lr_schedule(i // num_train_batches)
  
  opt_init, opt_update, get_params = optimizers.adam(step_size=lr_schedule, eps=1e-4)

  def batch_loss_fn(rng, params, images, beta):
    rngs = random.split(rng, batch_size)
    elbos, _, _, _ = jax.vmap(model.run, in_axes=(None, 0, 0, None))(params, images, rngs, beta)
    return -jnp.mean(elbos)

  @jit
  def run_epoch(epoch, rng, opt_state):
    beta = jnp.minimum(epoch / kl_threshold, 1.) if kl_annealing else 1.

    def body_fn(opt_state, args):
      idx, rng, batch = args
      loss, g = jax.value_and_grad(batch_loss_fn, argnums=1)(rng, get_params(opt_state), batch, beta)
      return opt_update(idx, g, opt_state), loss

    idxs = epoch * num_train_batches + jnp.arange(num_train_batches)
    rngs = random.split(rng, num_train_batches)
    scan_args = (idxs, rngs, train_batches)

    opt_state, losses = jax.lax.scan(body_fn, opt_state, scan_args)
    elbo = -jnp.mean(losses)

    return opt_state, elbo

  train_elbos = []
  test_elbos = []

  init_rng = random.PRNGKey(hps.init_seed)
  init_params = model.init_params(init_rng)
  opt_state = opt_init(init_params)

  num_worse = 0
  best_test_elbo = -1e20

  with trange(1, num_epochs+1) as t:
    for epoch in t:
      epoch_rng = random.PRNGKey(epoch)
      opt_state, train_elbo = run_epoch(epoch-1, epoch_rng, opt_state)
      train_elbos.append((epoch, float(train_elbo)))
      t.set_postfix(train_elbo=train_elbo)

      if save_epoch > 0 and epoch % save_epoch == 0:
        file_name = "{}/{:05d}.pkl".format(save_dir, epoch)
        utils.save_params(file_name, get_params(opt_state))

      if epoch % display_epoch == 0:
        test_elbo = dataset_elbo(model, test_batches, epoch_rng, get_params(opt_state))
        print("Epoch {} - Train {}, Test {}".format(epoch, train_elbo, test_elbo))

        test_elbo = float(test_elbo)
        test_elbos.append((epoch, test_elbo))
        
        if early_stopping and (not kl_annealing or epoch >= kl_threshold):
          if best_test_elbo > test_elbo + es_epsilon:
            num_worse += 1
            if num_worse >= patience:
              print("Early stopping at Epoch", epoch)
              break
          else:
            best_test_elbo = test_elbo
            num_worse = 0
  
  params = get_params(opt_state)
  file_name = "{}/params.pkl".format(save_dir)
  utils.save_params(file_name, params)
  print("Saved final params to", file_name)

  utils.save_params(save_dir + "/train_elbos.pkl", train_elbos)
  utils.save_params(save_dir + "/test_elbos.pkl", test_elbos)
  
  return params, train_elbos, test_elbos
