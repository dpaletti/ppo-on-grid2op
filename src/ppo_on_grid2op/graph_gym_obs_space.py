import numpy as np
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.Observation import BaseObservation


class GraphGymObsSpace(BoxGymObsSpace):  # type: ignore
    def __init__(
        self,
        n_nodes,
        n_edges,
        node_feature_space_size,
        edge_feature_space_size,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.node_feature_space_size = node_feature_space_size
        self.edge_feature_space_size = edge_feature_space_size
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        self.template_obs_vector = np.zeros(
            1  # current number of nodes in the graph
            + 1  # current number of edges
            + 1  # max number of nodes
            + 1  # max number of edges
            + 1  # node feature space size
            + 1  # edge feature space size
            + n_nodes
            * node_feature_space_size  # node features, select only 'number of nodes'
            + n_edges
            * 2  # edge connectivity such as edge_i connects (node_a, node_b), select only 'number of edges' pairs
            + n_edges
            * edge_feature_space_size  # edge features, select only number of edges
        )

        self._shape = self.template_obs_vector.shape  # overriding underlying Box shape

    def to_gym(self, grid2op_observation: BaseObservation) -> np.ndarray:
        """Convert a grid2op observation into a flattened graph.

        Args:
            grid2op_observation (BaseObservation): grid2op observation

        Returns:
            np.ndarray: a graph flattened into a vector
        """
        graph = grid2op_observation.get_energy_graph()
        node_features = []
        for node in graph:
            node_features.append([val.item() for val in graph.nodes[node].values()])

        node_features_matrix = GraphGymObsSpace._normalize_feature_matrix(
            np.stack(node_features)
        )

        edge_features = []
        for edge in graph.edges:
            edge_features.append([float(val) for val in graph.edges[edge].values()])
        if len(edge_features) > 0:
            edge_features_matrix = GraphGymObsSpace._normalize_feature_matrix(
                np.stack(edge_features)
            )
            edge_connectivity_matrix = np.stack(graph.edges)
        else:
            # End of episode case (only one node, no edges)
            edge_features_matrix = np.empty((0, 0))
            edge_connectivity_matrix = np.empty((0, 0))
        return self._flatten_graph_to_template(
            node_features_matrix, edge_features_matrix, edge_connectivity_matrix
        )

    @staticmethod
    def _normalize_feature_matrix(m: np.ndarray) -> np.ndarray:
        means = m.mean(0)
        ptp_vals = np.ptp(m, 0)

        result = np.zeros_like(m)

        mask = ptp_vals != 0
        result[:, mask] = (m[:, mask] - means[mask]) / ptp_vals[mask]

        # Columns with no variation become 0 (already initialized)
        return result

    def _flatten_graph_to_template(
        self,
        node_features_matrix: np.ndarray,
        edge_features_matrix: np.ndarray,
        edge_connectivity: np.ndarray,
    ) -> np.ndarray:
        current_n_nodes = node_features_matrix.shape[0]
        current_n_edges = edge_features_matrix.shape[0]
        output_obs = self.template_obs_vector.copy()

        if current_n_nodes == 1:
            # end of episode
            return output_obs

        # save current number of nodes and edges
        output_obs[0] = current_n_nodes
        output_obs[1] = current_n_edges

        # save max num nodes and edges
        output_obs[2] = self.n_nodes
        output_obs[3] = self.n_edges

        # save feature space size
        output_obs[4] = self.node_feature_space_size
        output_obs[5] = self.edge_feature_space_size

        start_idx = 6
        end_idx_current = start_idx + current_n_nodes * self.node_feature_space_size
        end_idx_fixed = start_idx + self.n_nodes * self.node_feature_space_size
        # save current node features in first array slot
        output_obs[start_idx:end_idx_current] = np.hstack(node_features_matrix)  # type: ignore

        start_idx = end_idx_fixed
        end_idx_current = end_idx_fixed + current_n_edges * 2
        end_idx_fixed = end_idx_fixed + self.n_edges * 2
        # save edge connectivity in second array slot
        output_obs[start_idx:end_idx_current] = np.hstack(edge_connectivity)  # type: ignore

        start_idx = end_idx_fixed
        end_idx_current = end_idx_fixed + current_n_edges * self.edge_feature_space_size
        end_idx_fixed = end_idx_fixed + self.n_edges * self.edge_feature_space_size

        # save edge features in third array slot
        output_obs[start_idx:end_idx_current] = np.hstack(edge_features_matrix)  # type: ignore

        return output_obs

    def close(self):
        pass
