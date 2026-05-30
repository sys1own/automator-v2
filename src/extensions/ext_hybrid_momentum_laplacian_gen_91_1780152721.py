# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 91
# Archetype: Velocity Momentum Tracking Layer
# Synthesis: Stochastic Hybrid (momentum_dampener x laplacian_governor via max)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 91,
    "purpose": "Velocity Momentum Tracking Layer",
    "compiled_timestamp": 1780152721.2185016,
    "hyper_parameters": {
        "regime": "Stochastic Hybrid (momentum_dampener x laplacian_governor via max)",
        "primary_archetype": "momentum_dampener",
        "secondary_archetype": "laplacian_governor",
        "blend_op": "max",
        "nonlinearity": "tanh+softsign",
        "reward_modulation": "reward_jax",
        "edge_weight": 1.388638,
        "safety_ratio": 0.666875,
        "factor_ceiling": 5.007723,
        "step_scale": 0.413954,
        "threshold_bias": 0.801319
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
        factor_a = jnp.clip(jnp.abs(jnp.tanh(1.0 / (hessian_factor_a + 1e-8))), 0.05, 5.008)
        lapvec_factor_b = jnp.array([v_val, -v_val, r_val, -r_val], dtype=jnp.float32)
        lapvec_factor_b = lapvec_factor_b / (jnp.linalg.norm(lapvec_factor_b) + 1e-8)
        lapstep_factor_b = lapvec_factor_b - 0.5 * (jnp.roll(lapvec_factor_b, -1) + jnp.roll(lapvec_factor_b, 1)) * 1.3886
        ritz_factor_b = jnp.dot(lapvec_factor_b, lapstep_factor_b) / (jnp.dot(lapvec_factor_b, lapvec_factor_b) + 1e-8)
        factor_b = jnp.clip(jnp.abs(((ritz_factor_b) / (1.0 + jnp.abs(ritz_factor_b)))), 0.05, 5.008)
        factor = jnp.clip(jnp.maximum(factor_a, factor_b), 0.01, 5.008)
        step_delta = factor * lr_val * v_val * reward_jax * 0.4140
        ceiling = float(np.maximum(12.0, jnp.abs(v_val) * 2.585))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
