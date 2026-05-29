import jax.numpy as jnp
import numpy as np

def execute_extension_pass(v, reward, lr):
    try:
        decay = jnp.exp(-jnp.abs(v - reward) * 0.046528)
        return float(np.clip(v * decay + (reward * lr * 0.1), -25.0, 25.0))
    except Exception: return v
