# pylint:disable=signature-differs, not-callable

import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.optim import Optimizer


class GraphTraversalOptimizer(Optimizer):
    """gradient free"""
    def __init__(self, params, lr=1e-3, num_directions=20, exploration=0.25,
                 update_interval=10, projection_interval=50, max_edges=100):
        params = list(params)
        defaults = dict(lr=lr, exploration=exploration)
        super().__init__(params, defaults)

        self.num_directions = num_directions
        self.update_interval = update_interval
        self.projection_interval = projection_interval
        self.max_edges = max_edges
        self.step_count = 0

        # Initialize parameter tracking
        self._param_groups = params
        self.param_list = []
        for group in self.param_groups:
            self.param_list.extend(group['params'])

        self.n_params = len(self.param_list)
        self.adj_matrix = sp.lil_matrix((self.n_params, self.n_params), dtype=np.float32)
        self.cov_matrix = defaultdict(lambda: defaultdict(float))
        self.projected_graph = None

        # Initialize perturbations
        self._init_perturbations()

    def _init_perturbations(self):
        self.perturbations = []
        for _ in range(self.num_directions):
            direction = []
            for p in self.param_list:
                direction.append(torch.randn_like(p) * self.defaults['lr'])
            self.perturbations.append(direction)

    def _project_graph(self):
        # Early exit for insufficient parameters
        if self.n_params < 2:
            self.projected_graph = defaultdict(set)
            return

        # Convert to dense matrix and normalize
        adj_dense = self.adj_matrix.toarray()
        np.fill_diagonal(adj_dense, 0)

        # Handle empty graph case
        if adj_dense.max() == 0:
            adj_dense = np.eye(self.n_params)  # Fallback to identity matrix
        else:
            adj_dense /= adj_dense.max()  # Normalize to [0,1]

        distance_matrix = 1 - adj_dense

        # Dynamic cluster calculation with bounds checking
        base_clusters = max(2, int(math.sqrt(self.n_params)))
        n_clusters = min(base_clusters, self.n_params - 1)

        # Final validation
        if n_clusters < 1 or n_clusters >= self.n_params:
            n_clusters = max(1, min(2, self.n_params - 1))

        # Safe clustering execution
        try:
            cluster = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            labels = cluster.fit_predict(distance_matrix)
        except Exception as e:
            print(f"Graph projection failed: {e}")
            labels = np.zeros(self.n_params, dtype=int)

        # Build projected graph with bounds checking
        self.projected_graph = defaultdict(set)
        for i in range(self.n_params):
            cluster_i = labels[i]
            valid_neighbors = [j for j in self.adj_matrix.rows[i] if j < self.n_params] # type:ignore
            for j in valid_neighbors:
                cluster_j = labels[j]
                if cluster_i != cluster_j and cluster_j < n_clusters:
                    self.projected_graph[cluster_i].add(cluster_j)

    def _get_candidate_edges(self):
        candidates = []

        # Existing strong edges
        if self.projected_graph is not None:
            for i in range(self.n_params):
                if i in self.projected_graph:
                    for j in self.projected_graph[i]:
                        if j < self.n_params:
                            candidates.append((i, j))
                            if len(candidates) >= self.max_edges:
                                return candidates

        # Fallback to random exploration
        n_random = min(self.max_edges, self.n_params*(self.n_params-1)//2)
        indices = np.random.choice(self.n_params, size=(n_random, 2), replace=False)
        for i, j in indices:
            if i != j:
                candidates.append((i, j))
        return candidates

    @torch.no_grad
    def step(self, closure):
        self.step_count += 1
        current_loss = closure(False)
        best_loss = current_loss
        best_params = [p.detach().clone() for p in self.param_list]

        # Generate candidate directions
        candidates = []

        # Individual parameter perturbations
        for _ in range(int(self.num_directions * self.defaults['exploration'])):
            idx = np.random.randint(self.n_params)
            direction = []
            for i, p in enumerate(self.param_list):
                if i == idx:
                    direction.append(torch.randn_like(p) * self.defaults['lr'])
                else:
                    direction.append(torch.zeros_like(p))
            candidates.append(direction)

        # Pairwise perturbations from graph
        edges = self._get_candidate_edges()
        for i, j in edges[:self.num_directions - len(candidates)]:
            direction = []
            for idx, p in enumerate(self.param_list):
                if idx == i or idx == j:
                    direction.append(torch.randn_like(p) * self.defaults['lr'])
                else:
                    direction.append(torch.zeros_like(p))
            candidates.append(direction)

        # Evaluate candidates
        for direction in candidates:
            original_params = [p.detach().clone() for p in self.param_list]

            # Apply perturbation
            for p, d in zip(self.param_list, direction):
                p.add_(d)

            # Evaluate loss
            with torch.no_grad():
                loss = closure(False)

            # Track improvements
            if loss < best_loss:
                best_loss = loss
                best_params = [p.detach().clone() for p in self.param_list]

                # Update covariance matrix
                active_params = [i for i, d in enumerate(direction) if d.abs().sum() > 0]
                for i in active_params:
                    for j in active_params:
                        if i != j:
                            self.cov_matrix[i][j] += 1

            # Restore parameters
            for p, orig in zip(self.param_list, original_params):
                p.copy_(orig)

        # Update parameters if improvement found
        if best_loss < current_loss:
            for p, best in zip(self.param_list, best_params):
                p.copy_(best)

        # Update graph structures periodically
        if self.step_count % self.update_interval == 0:
            # Update adjacency matrix from covariance
            for i in self.cov_matrix:
                total = sum(self.cov_matrix[i].values())
                for j in self.cov_matrix[i]:
                    weight = self.cov_matrix[i][j] / total
                    self.adj_matrix[i, j] = weight

            # Threshold edges
            self.adj_matrix[self.adj_matrix < 0.1] = 0

            # Convert to CSR to eliminate zeros and prune matrix
            self.adj_matrix = self.adj_matrix.tocsr()
            self.adj_matrix.eliminate_zeros()
            self.adj_matrix = self.adj_matrix.tolil()  # Convert back to LIL for future updates

            # Reset covariance
            self.cov_matrix.clear()

        # Project graph to lower dimension
        if self.step_count % self.projection_interval == 0:
            self._project_graph()

        return best_loss