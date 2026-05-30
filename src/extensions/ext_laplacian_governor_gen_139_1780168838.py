# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 139
# Archetype: Spectral Router Topology Controller
# Synthesis: Stochastic Pure (laplacian_governor)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 139,
    "purpose": "Spectral Router Topology Controller",
    "compiled_timestamp": 1780168838.3940344,
    "hyper_parameters": {
        "regime": "Stochastic Pure (laplacian_governor)",
        "primary_archetype": "laplacian_governor",
        "secondary_archetype": "None",
        "blend_op": "none",
        "nonlinearity": "sin",
        "reward_modulation": "jnp.abs(jnp.sin(reward_jax))",
        "edge_weight": 1.075479,
        "safety_ratio": 0.643533,
        "factor_ceiling": 2.008779,
        "step_scale": 0.430963,
        "threshold_bias": 0.773538
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
        lapvec_factor_a = jnp.array([v_val, -v_val, r_val, -r_val], dtype=jnp.float32)
        lapvec_factor_a = lapvec_factor_a / (jnp.linalg.norm(lapvec_factor_a) + 1e-8)
        lapstep_factor_a = lapvec_factor_a - 0.5 * (jnp.roll(lapvec_factor_a, -1) + jnp.roll(lapvec_factor_a, 1)) * 1.0755
        ritz_factor_a = jnp.dot(lapvec_factor_a, lapstep_factor_a) / (jnp.dot(lapvec_factor_a, lapvec_factor_a) + 1e-8)
        factor_a = jnp.clip(jnp.abs(jnp.sin(ritz_factor_a)), 0.05, 2.009)
        factor = jnp.clip(factor_a, 0.01, 2.009)
        step_delta = factor * lr_val * v_val * jnp.abs(jnp.sin(reward_jax)) * 0.4310
        ceiling = float(np.maximum(12.0, jnp.abs(v_val) * 1.527))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
