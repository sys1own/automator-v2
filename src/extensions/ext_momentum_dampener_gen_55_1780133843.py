# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 55
# Archetype: Velocity Momentum Tracking Layer
# Target Subsystem: Adaptive Meta-Gradient Step Scaling
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 55,
    "purpose": "Velocity Momentum Tracking Layer",
    "compiled_timestamp": 1780133843.9418478,
    "hyper_parameters": {
        "regime": "Implicit-Function Meta-Gradient Pacing",
        "differentiation_mode": "mixed_flow",
        "dynamic_bounds": True
    }
}

def verify_system_state(v, reward):
    if np.isnan(v) or np.isinf(v): return False
    if np.isnan(reward) or np.isinf(reward): return False
    return True

def execute_extension_pass(v, reward, lr):
    if not verify_system_state(v, reward): return v
    try:
        v_jax = jnp.array(float(v), dtype=jnp.float32)
        rew_jax = jnp.array(float(reward), dtype=jnp.float32)
        lr_jax = jnp.array(float(lr), dtype=jnp.float32)
        
        tracking_error = jnp.abs(v_jax - rew_jax)
        dynamic_regularizer = jnp.maximum(1e-5, tracking_error * lr_jax)
        
        grad_inner = v_jax - rew_jax
        hessian_inner = 1.0 + dynamic_regularizer * jnp.square(grad_inner)
        mixed_partial = -lr_jax * grad_inner
        
        best_response_jacobian = mixed_partial / (hessian_inner + 1e-8)
        step_delta = -best_response_jacobian * lr_jax * (1.0 + jnp.tanh(tracking_error))
        raw_output = v_jax + step_delta
        
        ceiling = float(jnp.maximum(25.0, jnp.abs(rew_jax) * 100.0))
        return float(np.clip(raw_output, 0.1, ceiling))
    except Exception: return v
