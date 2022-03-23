from jax import numpy as jnp
from jax import random

def hamiltonian_monte_carlo(
    current_q, current_p,
    q, p, epsilon,
    accept_trace, trace_len,
    U, K
):
    # Compute current and proposed Hamiltonians.
    current_H = K(current_p) + U(current_q)
    proposed_H = K(p) + U(q)
    
    # Metropolis accept-reject step.
    prob = jnp.exp(current_H - proposed_H)
    u = random.unif(prob.shape)
    accepts = prob > u
    q = q*accepts + current_q*(1.-accepts)
    
    accept_trace = accept_trace + accepts
    criteria = (accept_trace / trace_len) > 0.65
    adapt = 1.02*criteria + 0.98*(1. - criteria)
    epsilon = epsilon*adapt  # Missing clamp
    
    return q, epsilon, accept_trace
    

def leapfrog_integrator(
    current_q,  # Expect iid N(0, 1).
    current_p, 
    U, grad_U, 
    epsilon, num_steps=10
):
    epsilon = jnp.reshape(epsilon, newshape=(-1, 1))
    q = current_q
    
    # Start with half step for momentum.
    p = current_p - 0.5*epsilon*grad_U(q)
    
    for i in range(1, num_steps+1):
        q = q + p*epsilon  # Full step for position.
        
        # Full step for momentum, while not end of trajectory.
        if i < num_steps:
            p = p - epsilon*grad_U(q)
    
    # Finally do a half step for momentum.
    p = p - 0.5*epsilon*grad_U(q)
    
    # To make proposal symmetric.
    p = -p
    
    return q, p