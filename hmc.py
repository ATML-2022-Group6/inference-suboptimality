from jax import numpy as jnp
from jax import random
    
def leapfrog_integrator(current_q, current_p, U, grad_U, epsilon, num_steps=10):
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