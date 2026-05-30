# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 60
# Archetype: Shared Memory Boundary Filter
# Synthesis: Stochastic Pure (telemetry_compactor)
# ==============================================================================
import jax.numpy as jnp
import numpy as np

MODULE_METADATA = {
    "generation_id": 60,
    "purpose": "Shared Memory Boundary Filter",
    "compiled_timestamp": 1780139574.9680064,
    "hyper_parameters": {
        "regime": "Stochastic Pure (telemetry_compactor)",
        "primary_archetype": "telemetry_compactor",
        "secondary_archetype": "None",
        "blend_op": "none",
        "nonlinearity": "gauss",
        "reward_modulation": "(jnp.tanh(reward_jax) + reward_jax)",
        "edge_weight": 0.764316,
        "safety_ratio": 1.468235,
        "factor_ceiling": 4.584396,
        "step_scale": 0.189553,
        "threshold_bias": 0.950693
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
        p1_factor_a = 1.0 / (1.0 + jnp.exp(-jnp.clip(v_jax, -10.0, 10.0)))
        belief_factor_a = jnp.array([p1_factor_a, 1.0 - p1_factor_a])
        shannon_factor_a = -jnp.sum(belief_factor_a * jnp.log(belief_factor_a + 1e-12))
        dob_factor_a = 1.0 - (jnp.min(belief_factor_a) / (jnp.max(belief_factor_a) + 1e-12))
        factor_a = jnp.clip(jnp.abs(jnp.exp(-jnp.square(jnp.clip(shannon_factor_a * dob_factor_a, -3.0, 3.0)))) * 1.4682, 0.01, 4.584)
        factor = jnp.clip(factor_a, 0.01, 4.584)

        # STAGE 3: Relative step scaling (reward as directional variance amplifier)
        step_delta = factor * lr_val * v_val * (jnp.tanh(reward_jax) + reward_jax) * 0.1896
        ceiling = float(np.maximum(25.0, jnp.abs(v_val) * 1.725))
        return float(np.clip(v_val + float(step_delta), 0.1, ceiling))
    except Exception: return v
