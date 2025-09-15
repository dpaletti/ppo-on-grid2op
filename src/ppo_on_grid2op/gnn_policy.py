from typing import Any

import torch as th
import torch.nn as nn
import torch_geometric.nn as geom_nn
from grid2op.gym_compat import BoxGymObsSpace
from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy
from torch_geometric.data import Batch, Data


class ActorCriticGNNPolicy(ActorCriticPolicy):
    """
    Custom policy that uses GNN feature extraction.
    """

    def __init__(self, *args, **kwargs):
        custom_kwargs, kwargs = GraphFeaturesExtractor.prepare_init_args(kwargs)
        super().__init__(
            *args,
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=custom_kwargs,
            **kwargs,
        )


class MaskableActorCriticGNNPolicy(MaskableActorCriticPolicy):
    """
    Custom policy that uses GNN feature extraction.
    """

    def __init__(self, *args, **kwargs):
        custom_kwargs, kwargs = GraphFeaturesExtractor.prepare_init_args(kwargs)
        super().__init__(
            *args,
            features_extractor_class=GraphFeaturesExtractor,
            **kwargs,
        )


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: BoxGymObsSpace,  # type:ignore
        features_dim: int = 256,
        hidden_dim: int = 128,
        node_features: int = 8,
        edge_features: int = 27,
        dropout_p: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        # Determine input dimensions from observation space
        self.node_feature_dim = node_features
        self.edge_feature_dim = edge_features
        self.observation_space = observation_space

        self.gat_conv = geom_nn.GATv2Conv(
            self.node_feature_dim, hidden_dim, edge_dim=self.edge_feature_dim
        )

        self.batch_norm_nodes = geom_nn.BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.activation1 = nn.ReLU()

        # MLP layers after pooling
        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * 2, features_dim
            ),  # *2 because we concatenate mean and max pooling
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process observations through GNN.

        Args:
            observations: Tensor containing graph observations

        Returns:
            Processed features tensor
        """
        batch = self.batch_observations(observations)

        x = self.gat_conv(batch.x, batch.edge_index, batch.edge_attr)
        x = self.batch_norm_nodes(x)
        x = self.dropout(x)
        x = self.activation1(x)

        mean_pool = geom_nn.global_mean_pool(x, batch.batch)
        max_pool = geom_nn.global_max_pool(x, batch.batch)

        graph_features = th.cat([mean_pool, max_pool], dim=-1)
        return self.mlp(graph_features)

    def batch_observations(
        self,
        observations: th.Tensor,
    ) -> Batch:
        data_list = []
        for obs_index in range(len(observations)):
            node_features, edge_connections, edge_features = self.unpack_observation(
                observations[obs_index]
            )
            # Convert edge_connections to COO format
            edge_index = edge_connections.long().t().contiguous()

            data = Data(
                x=th.nan_to_num(node_features),
                edge_index=edge_index,
                edge_attr=th.nan_to_num(edge_features),
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        return batch

    def unpack_observation(
        self,
        observation: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        current_num_nodes = observation[0].int()
        current_num_edges = observation[1].int()

        max_num_nodes = observation[2].int()
        max_num_edges = observation[3].int()

        node_feature_space_size = observation[4].int()
        edge_feature_space_size = observation[5].int()

        start_idx = 6

        end_idx_current = start_idx + current_num_nodes * node_feature_space_size
        end_idx_fixed = start_idx + max_num_nodes * node_feature_space_size

        node_features = observation[start_idx:end_idx_current].reshape(
            (current_num_nodes, node_feature_space_size)
        )  # type:ignore

        start_idx = end_idx_fixed
        end_idx_current = end_idx_fixed + current_num_edges * 2
        end_idx_fixed = end_idx_fixed + max_num_edges * 2

        edge_connections = observation[start_idx:end_idx_current].reshape(
            (current_num_edges, 2)
        )  # type:ignore

        start_idx = end_idx_fixed
        end_idx_current = end_idx_fixed + current_num_edges * edge_feature_space_size
        end_idx_fixed = end_idx_fixed + max_num_edges * edge_feature_space_size

        edge_features = observation[start_idx:end_idx_current].reshape(
            (current_num_edges, edge_feature_space_size)
        )  # type:ignore

        return node_features, edge_connections, edge_features

    def prepare_init_args(kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        custom_kwargs = {}
        if kwargs.get("features_dim") is not None:
            custom_kwargs["features_dim"] = kwargs["features_dim"]
            del kwargs["features_dim"]

        if kwargs.get("hidden_dim") is not None:
            custom_kwargs["hidden_dim"] = kwargs["hidden_dim"]
            del kwargs["hidden_dim"]

        if kwargs.get("dropout_p") is not None:
            custom_kwargs["dropout_p"] = kwargs["dropout_p"]
            del kwargs["dropout_p"]

        if kwargs.get("features_extractor_kwargs") is not None:
            del kwargs["features_extractor_kwargs"]
        return custom_kwargs, kwargs
