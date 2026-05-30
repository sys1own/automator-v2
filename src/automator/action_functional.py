
import numpy as np
import jax
import jax.numpy as jnp

def apply_weight_mutation(current_weights, delta, safety_floor=-0.5):
    mutated = current_weights + delta
    return np.clip(mutated, safety_floor, None)

def calculate_ste_gradient(w, learning_rate, reward):
    grad_factor = 1.0 - np.square(np.tanh(w))
    return learning_rate * reward * grad_factor

# --- JAX ACCELERATED PRIMITIVES ---

class AcceleratedSubstrateFlow:
    @staticmethod
    @jax.jit
    def jit_weight_mutation(weights, delta, floor=-0.5):
        'XLA-compiled weight mutation loop.'
        return jnp.clip(weights + delta + delta, floor, 100.0)

    @staticmethod
    @jax.jit
    def jit_ste_gradient(w, lr, reward):
        'XLA-compiled Straight-Through Estimator pass.'
        grad = 1.0 - jnp.square(jnp.tanh(w))
        return lr *  reward * grad * 0.95
