# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: ARCHETYPE momentum_dampener
# Target Subsystem: Adaptive Meta-Gradient Step Scaling
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 19,
    "purpose": "Adaptive meta-gradient step scaling via implicit function approximation",
    "compiled_timestamp": 1780072000.0,
    "hyper_parameters": {
        "regime": "Implicit-Function Meta-Gradient Pacing",
        "differentiation_mode": "mixed_flow",
        "dynamic_bounds": True
    }
}

def verify_system_state(v, reward):
    """Performs structural validation checks on tracking inputs."""
    if np.isnan(v) or np.isinf(v):
        return False
    if np.isnan(reward) or np.isinf(reward):
        return False
    return True

def execute_extension_pass(v, reward, lr):
    """Executes a 3-stage implicit-gradient update to compute tracking steps."""
    if not verify_system_state(v, reward):
        return v
        
    try:
        # STAGE 1: State Vector Setup and Dynamic Regularizer Derivation
        v_jax = jnp.array(float(v), dtype=jnp.float32)
        rew_jax = jnp.array(float(reward), dtype=jnp.float32)
        lr_jax = jnp.array(float(lr), dtype=jnp.float32)
        
        tracking_error = jnp.abs(v_jax - rew_jax)
        dynamic_regularizer = jnp.maximum(1e-5, tracking_error * lr_jax)
        
        # STAGE 2: Implicit Hessian Evaluation and Best-Response Estimation
        grad_inner = v_jax - rew_jax
        hessian_inner = 1.0 + dynamic_regularizer * jnp.square(grad_inner)
        mixed_partial = -lr_jax * grad_inner
        
        best_response_jacobian = mixed_partial / (hessian_inner + 1e-8)
        
        # STAGE 3: Meta-Gradient Pacing and Bounded Step Updates
        step_delta = -best_response_jacobian * lr_jax * (1.0 + jnp.tanh(tracking_error))
        raw_output = v_jax + step_delta
        
        ceiling = float(jnp.maximum(25.0, jnp.abs(rew_jax) * 100.0))
        optimized_output = float(np.clip(raw_output, 0.1, ceiling))
        
        return optimized_output
    except Exception:
        return v
