from jax import numpy as jnp
from jax import random

def hmc_sample_and_tune(
    rng,
    current_q,
    current_p,
    grad_U,
    U, K,
    tuning_params,
):  
    """Our HMC magic function."""
    # Hamiltonian dynamics to propose new (q*, p*).
    proposed_q, proposed_p = _leapfrog_integrator(current_q, current_p, grad_U, stepsize)

    # Accept-reject procedure on (q*, p*) & adaptive tuning.
    current_q, stepsize, accept_trace = _hmc_accept_reject_adapt(
        rng,
        current_q, current_p,
        proposed_q, proposed_p, 
        U, K,
        tuning_params
    )
    tuning_params_no_period = (stepsize, accept_trace)
    
    return current_q, tuning_params_no_period

def _leapfrog_integrator(
    current_q,
    # Initial momentum is always initialized as iid N(0, 1).
    # Only reason we need is because of the two-part implementation.
    current_p, 
    grad_U, 
    stepsize, num_steps=10
):
    """
    Based on Xuechen, which in turn is based on 
    Neal (https://arxiv.org/pdf/1206.1901.pdf).
    """
    # For some reason, without the reshape things break.
    stepsize = jnp.reshape(stepsize, newshape=(-1, 1))  
    
    q = current_q
    
    # Start with half step for momentum.
    p = current_p - 0.5*stepsize*grad_U(q)
    
    for i in range(1, num_steps+1):
        q = q + p*stepsize  # Full step for position.
        # Full step for momentum, while not end of trajectory.
        if i < num_steps:
            p = p - stepsize*grad_U(q)
    
    # Finally do a half step for momentum.
    p = p - 0.5*stepsize*grad_U(q)
    
    # To make proposal symmetric. Not actually needed since K(p) = K(-p),
    # and p will be replaced in the next iteration.
    p = -p
    
    return q, p

def _hmc_accept_reject_adapt(
    rng,
    current_q, current_p,
    q, p,
    U, K,
    tuning_params,
):
    """
    Hamiltonian Monte Carlo accept-reject procedure
    and adaptive tuning procedure.
    """
    # Compute current and proposed Hamiltonians.
    current_H = K(current_p) + U(current_q)
    proposed_H = K(p) + U(q)
    
    # Metropolis accept-reject step.
    prob = jnp.exp(current_H - proposed_H)
    u = random.uniform(rng, prob.shape)
    accepts = prob > u
    q = q*accepts + current_q*(1.-accepts)
    
    # HMC tuning procedure.
    (
      stepsize, 
      accept_trace, 
      trace_period,
    ) = tuning_params
    accept_trace = accept_trace + accepts
    optimal_acceptance_rate = 0.65  # cf pg. 29 in https://arxiv.org/pdf/1206.1901.pdf
    
    criteria = accept_trace/trace_period > optimal_acceptance_rate
    adapt = 1.02*criteria + 0.98*(1. - criteria)
    stepsize = stepsize * adapt  # Missing clamp onto [1e-4, .5] as in XC; maybe needed in practice?
    
    return q, stepsize, accept_trace