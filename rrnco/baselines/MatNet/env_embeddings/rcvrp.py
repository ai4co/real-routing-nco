import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict


class RVRPInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).
    Embed the following node features to the embedding space:
        - demand: demand of the customers
        - distance : distance between the nodes
    """

    def __init__(
        self, embed_dim, linear_bias=True, use_coords=False, use_polar_feats=False
    ):
        super(RVRPInitEmbedding, self).__init__()

        self.embed_dim = embed_dim
        self.use_coords = use_coords
        self.use_polar_feats = use_polar_feats
        if self.use_coords:
            depot_feat_dim = 2
            city_feat_dim = 3
            self.init_embed_depot = nn.Linear(depot_feat_dim, embed_dim, linear_bias)
        else:
            depot_feat_dim = 1
            city_feat_dim = 1
            self.depot_client_emb = nn.Embedding(2, embed_dim)

        self.init_embed = nn.Linear(city_feat_dim, embed_dim, linear_bias)
        self.row_combine_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
        self.col_combine_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)

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

        if self.use_coords:
            # [batch, 1, 2]-> [batch, 1, embed_dim]
            td["locs"] = td["locs"].type(torch.float32)
            depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]

            depot_embedding = self.init_embed_depot(depot)
            # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embed_dim]

            cities_feats = torch.cat((cities, td["demand"][..., None]), -1)
            if self.use_polar_feats:
                city_locs_centered = cities - depot
                dist_to_depot = torch.norm(city_locs_centered, p=2, dim=-1, keepdim=True)
                angle_to_depot = torch.atan2(
                    city_locs_centered[..., 1:], city_locs_centered[..., :1]
                )
                cities_feats = torch.cat(
                    [cities_feats, dist_to_depot, angle_to_depot], dim=-1
                )
            node_embeddings = self.init_embed(cities_feats)
        else:
            depot_embedding, clients_embedding = self.depot_client_emb(
                torch.LongTensor([0, 1]).to(distance.device)
            ).chunk(2, dim=0)
            depot_embedding = depot_embedding.unsqueeze(1).expand(b, -1, -1)
            node_embeddings = self.init_embed(td["demand"][..., None])
            node_embeddings = clients_embedding + node_embeddings

        out = torch.cat((depot_embedding, node_embeddings), -2)
        row_emb = self.row_combine_embed(torch.cat([row_emb, out], -1))
        col_emb = self.col_combine_embed(torch.cat([col_emb, out], -1))

        return row_emb, col_emb, distance
