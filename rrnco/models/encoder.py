from typing import Tuple, Union

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from tensordict import TensorDict
from torch import Tensor

from rrnco.models.env_embeddings import env_init_embedding
from rrnco.models.nn.attn_freenet import AttnFreeNet


class RRNetEncoder(AutoregressiveEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
        nab_type: Type of Neural Adaptive Bias to use ("gating", "naive", or "heuristic")
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = None,
        env_name: str = "rcvrp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn=None,
        moe_kwargs: dict = None,
        use_coords: bool = False,
        use_polar_feats: bool = False,
        nab_type: str = "gating",
    ):
        super(RRNetEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(
                env_name, {"embed_dim": embed_dim, **init_embedding_kwargs}
            )
            if init_embedding is None
            else init_embedding
        )
        if env_name == "atsp" or env_name == "rcvrp":
            use_duration_matrix = False
        else:
            use_duration_matrix = True
        self.net = (
            AttnFreeNet(
                embed_dim=embed_dim,
                feedforward_hidden=feedforward_hidden,
                num_layers=num_layers,
                normalization=normalization,
                use_duration_matrix=use_duration_matrix,
                nab_type=nab_type,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, phase: str, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """

        # Transfer to embedding space
        row_emb, col_emb, distance = self.init_embedding(td, phase)

        if self.env_name == "rcvrptw":
            # Process embedding
            row_emb, col_emb = self.net(
                row_emb,
                col_emb,
                distance,
                td["locs"].type(torch.float32),
                td["duration_matrix"].type(torch.float32),
            )
        else:
            row_emb, col_emb = self.net(
                row_emb, col_emb, distance, td["locs"].type(torch.float32)
            )
        # Return latent representation and initial embedding
        return row_emb, col_emb
