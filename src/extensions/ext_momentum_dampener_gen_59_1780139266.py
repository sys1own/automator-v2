# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 59
# Archetype: Velocity Momentum Tracking Layer
# Synthesis: Stochastic Pure (momentum_dampener)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 59,
    "purpose": "Velocity Momentum Tracking Layer",
    "compiled_timestamp": 1780139266.5602071,
    "hyper_parameters": {
        "regime": "Stochastic Pure (momentum_dampener)",
        "primary_archetype": "momentum_dampener",
        "secondary_archetype": "None",
        "blend_op": "none",
        "nonlinearity": "softsign",
        "reward_modulation": "(jnp.sin(reward_jax) + reward_jax)",
        "edge_weight": 0.562244,
        "safety_ratio": 0.649590,
        "factor_ceiling": 3.619735,
        "step_scale": 0.146058,
        "threshold_bias": 1.096580
    }
}

def verify_system_state(v, reward):
    if np.isnan(v) or np.isinf(v): return False
    if np.isnan(reward) or np.isinf(reward): return False
    return True

def execute_extension_pass(v, reward, lr):
    if not verify_system_state(v, reward): return v
    try:
        # STAGE 1: State vector setup (reward retained as a scale amplifier)
        v_val, r_val, lr_val = float(v), float(reward), float(lr)
        reward_jax = jnp.array(r_val, dtype=jnp.float32)
        v_jax = jnp.array(v_val, dtype=jnp.float32)

        # STAGE 2: Stochastically-synthesised factor computation
        trackerr_factor_a = jnp.abs(v_jax - reward_jax)
        regular_factor_a = jnp.maximum(1e-5, trackerr_factor_a * lr_val)
        hessian_factor_a = 1.0 + regular_factor_a * jnp.square(v_jax - reward_jax)
        factor_a = jnp.clip(jnp.abs(((1.0 / (hessian_factor_a + 1e-8)) / (1.0 + jnp.abs(1.0 / (hessian_factor_a + 1e-8))))), 0.05, 3.620)
        factor = jnp.clip(factor_a, 0.01, 3.620)

        # STAGE 3: Relative step scaling (reward as directional variance amplifier)
        step_delta = factor * lr_val * v_val * (jnp.sin(reward_jax) + reward_jax) * 0.1461
        ceiling = float(np.maximum(50.0, jnp.abs(v_val) * 2.601))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
