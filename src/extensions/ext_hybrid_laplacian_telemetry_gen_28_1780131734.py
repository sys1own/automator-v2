# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 28
# Archetype: Spectral Router Topology Controller
# Synthesis: Stochastic Hybrid (laplacian_governor x telemetry_compactor via max)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 28,
    "purpose": "Spectral Router Topology Controller",
    "compiled_timestamp": 1780131734.780576,
    "hyper_parameters": {
        "regime": "Stochastic Hybrid (laplacian_governor x telemetry_compactor via max)",
        "primary_archetype": "laplacian_governor",
        "secondary_archetype": "telemetry_compactor",
        "blend_op": "max",
        "nonlinearity": "tanh+sin",
        "reward_modulation": "jnp.abs(jnp.sin(reward_jax))",
        "edge_weight": 1.080288,
        "safety_ratio": 1.030937,
        "factor_ceiling": 2.992983,
        "step_scale": 0.399222,
        "threshold_bias": 1.141894
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
        lapvec_factor_a = jnp.array([v_val, -v_val, r_val, -r_val], dtype=jnp.float32)
        lapvec_factor_a = lapvec_factor_a / (jnp.linalg.norm(lapvec_factor_a) + 1e-8)
        lapstep_factor_a = lapvec_factor_a - 0.5 * (jnp.roll(lapvec_factor_a, -1) + jnp.roll(lapvec_factor_a, 1)) * 1.0803
        ritz_factor_a = jnp.dot(lapvec_factor_a, lapstep_factor_a) / (jnp.dot(lapvec_factor_a, lapvec_factor_a) + 1e-8)
        factor_a = jnp.clip(jnp.abs(jnp.tanh(ritz_factor_a)), 0.05, 2.993)
        p1_factor_b = 1.0 / (1.0 + jnp.exp(-jnp.clip(v_jax, -10.0, 10.0)))
        belief_factor_b = jnp.array([p1_factor_b, 1.0 - p1_factor_b])
        shannon_factor_b = -jnp.sum(belief_factor_b * jnp.log(belief_factor_b + 1e-12))
        dob_factor_b = 1.0 - (jnp.min(belief_factor_b) / (jnp.max(belief_factor_b) + 1e-12))
        factor_b = jnp.clip(jnp.abs(jnp.sin(shannon_factor_b * dob_factor_b)) * 1.0309, 0.01, 2.993)
        factor = jnp.clip(jnp.maximum(factor_a, factor_b), 0.01, 2.993)

        # STAGE 3: Relative step scaling (reward as directional variance amplifier)
        step_delta = factor * lr_val * v_val * jnp.abs(jnp.sin(reward_jax)) * 0.3992
        ceiling = float(np.maximum(12.0, jnp.abs(v_val) * 1.716))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
