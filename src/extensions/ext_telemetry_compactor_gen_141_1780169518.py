# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 141
# Archetype: Shared Memory Boundary Filter
# Synthesis: Stochastic Pure (telemetry_compactor)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 141,
    "purpose": "Shared Memory Boundary Filter",
    "compiled_timestamp": 1780169518.3227794,
    "hyper_parameters": {
        "regime": "Stochastic Pure (telemetry_compactor)",
        "primary_archetype": "telemetry_compactor",
        "secondary_archetype": "None",
        "blend_op": "none",
        "nonlinearity": "arctan",
        "reward_modulation": "(jnp.sin(reward_jax) + reward_jax)",
        "edge_weight": 0.411577,
        "safety_ratio": 1.247861,
        "factor_ceiling": 2.823896,
        "step_scale": 0.106806,
        "threshold_bias": 1.139131
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
        p1_factor_a = 1.0 / (1.0 + jnp.exp(-jnp.clip(v_jax, -10.0, 10.0)))
        belief_factor_a = jnp.array([p1_factor_a, 1.0 - p1_factor_a])
        shannon_factor_a = -jnp.sum(belief_factor_a * jnp.log(belief_factor_a + 1e-12))
        dob_factor_a = 1.0 - (jnp.min(belief_factor_a) / (jnp.max(belief_factor_a) + 1e-12))
        factor_a = jnp.clip(jnp.abs(jnp.arctan(shannon_factor_a * dob_factor_a)) * 1.2479, 0.01, 2.824)
        factor = jnp.clip(factor_a, 0.01, 2.824)
        step_delta = factor * lr_val * v_val * (jnp.sin(reward_jax) + reward_jax) * 0.1068
        ceiling = float(np.maximum(50.0, jnp.abs(v_val) * 2.337))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
