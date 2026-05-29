import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple, Optional

class SparseGraphTuple(NamedTuple):
    nodes: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray
    edge_weights: jnp.ndarray
    num_nodes: int
    num_edges: int

class JaxMatrixFreeSpectralRouter:
    def __init__(self, embedding_dim: int = 4, oversamples: int = 5, power_iters: int = 2):
        self.k = embedding_dim
        self.p = oversamples
        self.q = power_iters

    @partial(jax.jit, static_argnums=(0,))
    def _compute_laplacian_matvec(self, graph: SparseGraphTuple, v: jnp.ndarray) -> jnp.ndarray:
        # L_sym = I - D^{-1/2} A D^{-1/2}
        deg = jnp.zeros((graph.nodes.shape[0],), dtype=jnp.float32)
        deg = deg.at[graph.senders].add(graph.edge_weights)
        deg = deg.at[graph.receivers].add(graph.edge_weights)
        inv_sqrt_deg = jnp.where(deg > 0.0, jax.lax.rsqrt(deg), 0.0)
        
        scaled_v = v * inv_sqrt_deg[:, None]
        messages = scaled_v[graph.senders] * graph.edge_weights[:, None]
        aggregated = jnp.zeros_like(v).at[graph.receivers].add(messages)
        aggregated = aggregated.at[graph.senders].add(scaled_v[graph.receivers] * graph.edge_weights[:, None])
        
        return v - (aggregated * inv_sqrt_deg[:, None])

    @partial(jax.jit, static_argnums=(0,))
    def route_frame(self, graph: SparseGraphTuple, key: jax.Array) -> jnp.ndarray:
        # n is derived from graph.nodes.shape[0] which JAX can trace as a static-sized buffer
        n = graph.nodes.shape[0]
        l_dim = self.k + self.p
        
        omega = jax.random.normal(key, shape=(n, l_dim), dtype=jnp.float32)
        y = omega
        for _ in range(self.q + 1):
            y = self._compute_laplacian_matvec(graph, y)
            
        q, _ = jnp.linalg.qr(y)
        l_q = self._compute_laplacian_matvec(graph, q)
        b = jnp.matmul(q.T, l_q)
        
        _, _, vt = jnp.linalg.svd(b, full_matrices=False)
        u_approx = jnp.matmul(q, vt.T)
        return u_approx[:, 1:self.k + 1]

class EquinoxGCNConv:
    def __init__(self, in_features: int, out_features: int, key: jax.Array):
        w_key, b_key = jax.random.split(key, 2)
        lim = jnp.sqrt(6.0 / (in_features + out_features))
        self.weight = jax.random.uniform(w_key, (in_features, out_features), minval=-lim, maxval=lim)
        self.bias = jnp.zeros((out_features,))

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, graph: SparseGraphTuple) -> SparseGraphTuple:
        deg = jnp.zeros((graph.nodes.shape[0],), dtype=jnp.float32)
        deg = deg.at[graph.senders].add(graph.edge_weights)
        deg = deg.at[graph.receivers].add(graph.edge_weights)
        norm_coef = jnp.where(deg > 0.0, jax.lax.rsqrt(deg), 0.0)
        
        h = jnp.matmul(graph.nodes, self.weight) + self.bias
        h_scaled = h * norm_coef[:, None]
        messages = h_scaled[graph.senders] * graph.edge_weights[:, None]
        aggregated = jnp.zeros_like(h).at[graph.receivers].add(messages)
        
        output_nodes = jax.nn.relu(aggregated * norm_coef[:, None])
        return graph._replace(nodes=output_nodes)