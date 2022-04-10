from jax import numpy as jnp
from jax import random, lax

def hmc_sample_and_tune(
  rng,
  current_q,
  current_p,
  U, K,
  grad_U,
  tuning_params: tuple,  # (stepsize, accept_trace, trace_period_elapsed)
):
  """Our HMC magic function."""
  stepsize, _, _ = tuning_params

  # Hamiltonian dynamics to propose new (q*, p*).
  proposed_q, proposed_p = _leapfrog_integrator(current_q, current_p,
                                                grad_U, stepsize)

  # Accept-reject procedure on (q*, p*) & adaptive tuning.
  proposed_q, tuned_stepsize, updated_accept_trace = _hmc_accept_reject_adapt(
    rng,
    current_q, current_p,
    proposed_q, proposed_p,
    U, K,
    tuning_params
  )
  return proposed_q, tuned_stepsize, updated_accept_trace

def _leapfrog_integrator(
  current_q,
  # Initial momentum p is always initialized as iid N(0, 1).
  # Only reason we need is because of the two-part implementation.
  current_p,
  grad_U,
  stepsize, num_steps=10
):
  """
  Based on Xuechen, which in turn is based on
  Neal (https://arxiv.org/pdf/1206.1901.pdf).
  """
  q = current_q

  # Start with half step for momentum.
  p = current_p - 0.5*stepsize*grad_U(q)

  def loop(acc, _):
    q, p = acc
    q += p * stepsize
    p -= stepsize * grad_U(q)
    return (q, p), None

  (q, p), _ = lax.scan(loop, (q, p), jnp.arange(num_steps-1))
  q += p * stepsize

  # for i in range(1, num_steps+1):
  #   q = q + p*stepsize  # Full step for position.
  #   # Full step for momentum, while not end of trajectory.
  #   if i < num_steps:
  #     p = p - stepsize*grad_U(q)

  # Finally do a half step for momentum.
  p = p - 0.5*stepsize*grad_U(q)

  # To make proposal symmetric. Not actually needed since K(p) = K(-p),
  # and p will be replaced in the next iteration.
  p = -p

  return q, p

def _hmc_accept_reject_adapt(
  rng,
  current_q, current_p,
  proposed_q, proposed_p,
  U, K,
  tuning_params: tuple,  # (stepsize, accept_trace, trace_period_elapsed)
):
  """
  Hamiltonian Monte Carlo accept-reject procedure
  and adaptive tuning procedure.
  """
  # Compute current and proposed Hamiltonians H(q, p).
  current_H = U(current_q) + K(current_p)
  proposed_H = U(proposed_q) + K(proposed_p)

  # Metropolis accept-reject step.
  prob = jnp.exp(current_H - proposed_H)
  u = random.uniform(rng, prob.shape)
  accepts = prob > u
  proposed_q = proposed_q*accepts + current_q*(1. - accepts)

  # HMC tuning procedure.
  (
    stepsize,
    accept_trace,
    trace_period_elapsed,
  ) = tuning_params
  optimal_acceptance_rate = 0.65  # cf. pg. 29 in Neal (https://arxiv.org/pdf/1206.1901.pdf)

  # Tuning step.
  updated_accept_trace = accept_trace + accepts
  criteria = (updated_accept_trace/trace_period_elapsed
                > optimal_acceptance_rate)
  adapt = 1.02*criteria + 0.98*(1. - criteria)
  tuned_stepsize = stepsize * adapt
  tuned_stepsize = lax.clamp(1e-4, tuned_stepsize, .5)

  return proposed_q, tuned_stepsize, updated_accept_trace