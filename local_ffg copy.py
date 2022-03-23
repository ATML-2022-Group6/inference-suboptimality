from vae import log_bernoulli #, log_mean_exp, safe_repeat
from jax.example_libraries import stax
import torch.optim as optim
from jax.scipy.special import logsumexp
import time
from jax.scipy import stats
from jax import numpy as jnp

from jax.example_libraries import optimizers
from tqdm.notebook import tqdm, trange
from jax import jit
from jax import random
import jax

num_samples = 32

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3, eps=1e-4)

def loss_fn(rng, enc_params, decoder_params, image):
  mu, logvar = enc_params
  elbo, _, _, _= run_vae_local(rng, image, mu, logvar, decoder_params)
  return -elbo

def iwelbo_fn(rng, enc_params, decoder_params, image):
    rngs = random.split(rng, num_samples)
    mu, logvar = enc_params
    iw_log_summand, _, _, _ = jax.vmap(run_vae_local, in_axes=(0, None, None, None, None))( rngs, image, mu, logvar, decoder_params )

    K = num_samples
    iwelbo_K = logsumexp(iw_log_summand) - jnp.log(K)
    return iwelbo_K

def batch_iwelbo_fn(rng, enc_params, decoder_params, images):
  rngs = random.split(rng, batch_size)
  return jnp.mean(jax.vmap(iwelbo_fn, in_axes=(0, 0, None, 0))(rngs, enc_params, decoder_params, images))


def batch_loss_fn(rng, enc_params, decoder_params, images):
  rngs = random.split(rng, batch_size)
  return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, None, 0))(rngs, enc_params, decoder_params, images))

@jit
def run_iwelbo_epoch(epoch, rng,opt_state, decoder_params, batch):

  def body_fn(opt_state, args):
    idx, rng, batch = args
    enc_params = get_params(opt_state)
    loss = batch_iwelbo_fn(rng, enc_params,decoder_params, batch)
    return loss

  scan_args = (epoch, rng, batch)
  losses = body_fn(opt_state, scan_args)

  return losses

@jit
def run_epoch(epoch, rng, opt_state, decoder_params, batch):

  def body_fn(opt_state, args):
    idx, rng, batch = args
    enc_params = get_params(opt_state)
    loss, g = jax.value_and_grad(batch_loss_fn, argnums=1)(rng, enc_params, decoder_params, batch)
    return opt_update(idx, g, opt_state), loss

  scan_args = (epoch, rng, batch)
  opt_state, losses = body_fn(opt_state, scan_args)

  return opt_state, losses

# def log_mean_exp(x):
#     if jnp.shape(x)[0] == 1:
#       return x[0][0]
#     max_, _ = jnp.max(x, 1, keepdims=True)
#     return jnp.log(jnp.mean(jnp.exp(x - max_), 1)) + jnp.squeeze(max_)

def optimize_local_gaussian(
    log_likelihood,
    decoder_params,
    batch,
    z_size,
    check_every=100,
    sentinel_thres=10,
    debug=False
):
    # init_rng = random.PRNGKey(0)
    # init_encoder_params, _ = init_vae(rng=init_rng, input_shape=(28 * 28,))
    batch_size = len(batch)
    latent_size = z_size
    mu0 = jnp.zeros((batch_size,latent_size))
    logvar0 = jnp.zeros((batch_size, latent_size))
    init_params = (mu0, logvar0)

    opt_state = opt_init(init_params)
    rng = random.PRNGKey(0)

    best_avg, sentinel, train_elbo, train_iwae = 999999, 0, [], []
    # perform local opt
    time_ = time.time()

    plot_elbo, plot_iwae = [], []

    for epoch in range(1, 999999):     
        rng, epoch_rng = random.split(rng)
        opt_state, loss = run_epoch(epoch-1, rng, opt_state, decoder_params, batch)
        
        iw_loss = run_iwelbo_epoch(epoch-1, rng,opt_state, decoder_params, batch)

        # if epoch % 1000 == 0:
        #   print(epoch, loss)
        train_elbo.append(loss)
        train_iwae.append(iw_loss)
        if epoch % check_every == (check_every-1):
            last_avg = jnp.mean(jnp.array(train_elbo))
            last_avg_iwae = jnp.mean(jnp.array(train_iwae))
            if debug:  # debugging helper
                sys.stderr.write(
                    'Epoch %d, time elapse %.4f, last avg %.4f, prev best %.4f\n' % \
                    (epoch, time.time()-time_, -last_avg, -best_avg)
                )
            if last_avg < best_avg:
                sentinel, best_avg = 0, last_avg
            else:
                sentinel += 1
            if sentinel > sentinel_thres:
                break
            train_elbo = []
            train_iwae = []
            plot_elbo.append(-last_avg)
            plot_iwae.append(last_avg_iwae)
            time_ = time.time()


    # evaluation
    vae_elbo = -loss
    iwae_elbo = iw_loss
    return vae_elbo, iwae_elbo, plot_elbo, plot_iwae

def local_FFG(params, z_size, batches):
    _, decoder_params = params
    vae_record, iwae_record = [], []
    time_ = time.time()
    prev_seq = []
    for i, batch in enumerate(tqdm(batches)):
        elbo, iwae, _ , _ = optimize_local_gaussian(log_bernoulli, decoder_params, batch, z_size)
        vae_record.append(elbo)
        iwae_record.append(iwae)
        print ('Local opt w/ ffg, batch %d, time elapse %.4f, ELBO %.4f, IWAE %.4f' % \
            (i+1, time.time()-time_, elbo, iwae))
        print ('mean of ELBO so far %.4f, mean of IWAE so far %.4f' % \
            (np.nanmean(vae_record), np.nanmean(iwae_record)))
        time_ = time.time()

    print ('Finishing...')
    print ('Average ELBO %.4f, IWAE %.4f' % (np.nanmean(vae_record), np.nanmean(iwae_record)))
    



#To run: 
# local_FFG(get_params(opt_state), 50, train_batches)