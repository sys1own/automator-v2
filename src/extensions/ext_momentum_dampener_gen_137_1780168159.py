# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 137
# Archetype: Velocity Momentum Tracking Layer
# Synthesis: Stochastic Pure (momentum_dampener)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 137,
    "purpose": "Velocity Momentum Tracking Layer",
    "compiled_timestamp": 1780168159.2965117,
    "hyper_parameters": {
        "regime": "Stochastic Pure (momentum_dampener)",
        "primary_archetype": "momentum_dampener",
        "secondary_archetype": "None",
        "blend_op": "none",
        "nonlinearity": "sin",
        "reward_modulation": "jnp.abs(jnp.sin(reward_jax))",
        "edge_weight": 0.671296,
        "safety_ratio": 1.038464,
        "factor_ceiling": 2.631419,
        "step_scale": 0.200786,
        "threshold_bias": 0.658346
    }
}

def verify_system_state(v, reward):
    if np.isnan(v) or np.isinf(v): return False
    if np.isnan(reward) or np.isinf(reward): return False
    return True

def execute_extension_pass(v, reward, lr):
    if not verify_system_state(v, reward): return v
    try:
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        reward_jax = jnp.array(r_val, dtype=jnp.float32)
        v_jax = jnp.array(v_val, dtype=jnp.float32)
        trackerr_factor_a = jnp.abs(v_jax - reward_jax)
        regular_factor_a = jnp.maximum(1e-5, trackerr_factor_a * lr_val)
        hessian_factor_a = 1.0 + regular_factor_a * jnp.square(v_jax - reward_jax)
        factor_a = jnp.clip(jnp.abs(jnp.sin(1.0 / (hessian_factor_a + 1e-8))), 0.05, 2.631)
        factor = jnp.clip(factor_a, 0.01, 2.631)
        step_delta = factor * lr_val * v_val * jnp.abs(jnp.sin(reward_jax)) * 0.2008
        ceiling = float(np.maximum(50.0, jnp.abs(v_val) * 2.009))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
