# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 63
# Archetype: Spectral Router Topology Controller
# Synthesis: Stochastic Hybrid (laplacian_governor x momentum_dampener via geom)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 63,
    "purpose": "Spectral Router Topology Controller",
    "compiled_timestamp": 1780142475.3505607,
    "hyper_parameters": {
        "regime": "Stochastic Hybrid (laplacian_governor x momentum_dampener via geom)",
        "primary_archetype": "laplacian_governor",
        "secondary_archetype": "momentum_dampener",
        "blend_op": "geom",
        "nonlinearity": "arctan+tanh",
        "reward_modulation": "(jnp.tanh(reward_jax) + reward_jax)",
        "edge_weight": 0.374934,
        "safety_ratio": 1.055410,
        "factor_ceiling": 5.071184,
        "step_scale": 0.386792,
        "threshold_bias": 1.456505
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
        lapstep_factor_a = lapvec_factor_a - 0.5 * (jnp.roll(lapvec_factor_a, -1) + jnp.roll(lapvec_factor_a, 1)) * 0.3749
        ritz_factor_a = jnp.dot(lapvec_factor_a, lapstep_factor_a) / (jnp.dot(lapvec_factor_a, lapvec_factor_a) + 1e-8)
        factor_a = jnp.clip(jnp.abs(jnp.arctan(ritz_factor_a)), 0.05, 5.071)
        trackerr_factor_b = jnp.abs(v_jax - reward_jax)
        regular_factor_b = jnp.maximum(1e-5, trackerr_factor_b * lr_val)
        hessian_factor_b = 1.0 + regular_factor_b * jnp.square(v_jax - reward_jax)
        factor_b = jnp.clip(jnp.abs(jnp.tanh(1.0 / (hessian_factor_b + 1e-8))), 0.05, 5.071)
        factor = jnp.clip(jnp.sqrt(jnp.abs(factor_a * factor_b) + 1e-8), 0.01, 5.071)
        step_delta = factor * lr_val * v_val * (jnp.tanh(reward_jax) + reward_jax) * 0.3868
        ceiling = float(np.maximum(50.0, jnp.abs(v_val) * 1.713))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
