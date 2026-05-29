# ==============================================================================
# AUTONOMOUS EXTENSION LAYER: GENERATION 17
# Archetype: Spectral Router Topology Controller
# Target Subsystem: Graph Spectral Router Boundary Constraints
# ==============================================================================
import jax.numpy as jnp
import numpy as np

ROUTING_DIMS = 4
POWER_ITER_STEPS = 5
TOPOLOGICAL_CONTRACTION_BOUND = 12.0

MODULE_METADATA = {
    "generation_id": 17,
    "purpose": "Spectral Router Topology Controller",
    "compiled_timestamp": 1780070100.0,
    "hyper_parameters": {
        "nodes": ROUTING_DIMS,
        "iterations": POWER_ITER_STEPS,
        "clipping_ceiling": TOPOLOGICAL_CONTRACTION_BOUND
    }
}

def execute_extension_pass(v, reward, lr):
    try:
        v_jax = jnp.array(v, dtype=jnp.float32)
        reward_jax = jnp.array(reward, dtype=jnp.float32)
        lr_jax = jnp.array(lr, dtype=jnp.float32)

        def compute_laplacian_matvec(vec):
            left_shift = jnp.roll(vec, shift=-1)
            right_shift = jnp.roll(vec, shift=1)
            adjacency_product = 0.5 * (left_shift + right_shift)
            return vec - adjacency_product

        seed_value = jnp.sin(reward_jax)
        initial_vector = jnp.array([seed_value, -seed_value, seed_value * 0.5, -seed_value * 0.5], dtype=jnp.float32)

        ones_vector = jnp.ones((ROUTING_DIMS,), dtype=jnp.float32)
        nullspace_projection = jnp.dot(initial_vector, ones_vector) * ones_vector / float(ROUTING_DIMS)
        orthogonal_vector = initial_vector - nullspace_projection
        orthogonal_vector = orthogonal_vector / (jnp.linalg.norm(orthogonal_vector) + 1e-8)

        current_eigenvector = orthogonal_vector
        for _ in range(POWER_ITER_STEPS):
            laplacian_step = compute_laplacian_matvec(current_eigenvector)
            iterated_vector = current_eigenvector - laplacian_step
            iterated_vector = iterated_vector - (jnp.dot(iterated_vector, ones_vector) * ones_vector / float(ROUTING_DIMS))
            vector_norm = jnp.linalg.norm(iterated_vector)
            current_eigenvector = jnp.where(vector_norm > 0.0, iterated_vector / vector_norm, current_eigenvector)

        num = jnp.dot(current_eigenvector, compute_laplacian_matvec(current_eigenvector))
        den = jnp.dot(current_eigenvector, current_eigenvector)
        algebraic_connectivity = jnp.where(den > 0.0, num / den, 0.5)

        contraction_pacing = jnp.clip(algebraic_connectivity, 0.1, 1.0)
        step_delta = contraction_pacing * lr_jax * (reward_jax - v_jax) * 0.25
        
        return float(jnp.clip(v_jax + step_delta, 0.1, TOPOLOGICAL_CONTRACTION_BOUND))
    except Exception:
        return v
