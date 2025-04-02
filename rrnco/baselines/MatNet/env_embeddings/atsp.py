import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict


class ATSPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - demand: demand of the customers
        - distance : distance between the nodes
    """

    def __init__(
        self, embed_dim, linear_bias=True, use_coords=False, use_polar_feats=False
    ):
        super(ATSPInitEmbedding, self).__init__()

        self.embed_dim = embed_dim

    def forward(self, td: TensorDict):
        distance = td["distance_matrix"]

        b, r, c = distance.shape
        row_emb = torch.zeros(b, r, self.embed_dim, device=distance.device)
        # MatNet uses one-hot encoding for column embeddings
        col_emb = torch.zeros(b, c, self.embed_dim, device=distance.device)
        rand = torch.rand(b, c)
        rand_idx = rand.argsort(dim=1)
        b_idx = torch.arange(b)[:, None].expand(b, c)
        n_idx = torch.arange(c)[None, :].expand(b, c)
        col_emb[b_idx, n_idx, rand_idx] = 1.0

        return row_emb, col_emb, distance
