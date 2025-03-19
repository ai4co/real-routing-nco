import torch
import torch.nn as nn


class RVRPTWInitEmbedding(nn.Module):
    """Initial embedding for the Vehicle Routing Problems (VRP).

    Embeds node features into a shared embedding space:
    - locs: x, y coordinates (for depot and customers)
    - demand: customer demands
    - distance: pairwise distances between nodes
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        use_coords=True,
        use_polar_feats=True,
        use_dist=True,
        use_matnet_init=True,
        sample_type="prob",
        sample_size=25,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_polar_feats = use_polar_feats
        self.use_dist = use_dist
        self.use_matnet_init = use_matnet_init
        self.embed_dim = embed_dim

        # Initialize embeddings based on configurations
        if not self.use_dist:
            self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
            self.init_embed = nn.Linear(4, embed_dim, linear_bias)
            if self.use_matnet_init:
                self.combine_row_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
                self.combine_col_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
        else:
            if self.use_coords:
                self.coord_expert = CoordinateExpert(3, embed_dim)
                self.gating_network_row = ContextualGating(embed_dim)
                self.gating_network_col = ContextualGating(embed_dim)
                self.init_embed = nn.Linear(4, embed_dim, linear_bias)
                self.combine_row_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
                self.combine_col_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
            self.distance_expert = DistanceExpert(embed_dim, sample_type, sample_size)

    def forward(self, td, phase):
        locs = td["locs"].float()
        demand = td["demand_linehaul"]
        time_windows = td["time_windows"]
        service_time = td["service_time"]
        vrp_attr = torch.cat(
            [demand.unsqueeze(-1), time_windows, service_time.unsqueeze(-1)], dim=-1
        )
        distance = td["distance_matrix"]

        if not self.use_dist:
            return self._embed_without_distance(locs, vrp_attr, distance)
        return self._embed_with_distance(locs, vrp_attr, distance, phase)

    def _embed_without_distance(self, locs, demand, distance):
        depot, cities = locs[:, :1, :], locs[:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        cities_feats = torch.cat([cities, demand[..., None]], dim=-1)

        if self.use_polar_feats:
            city_locs_centered = cities - depot
            angle_to_depot = torch.atan2(
                city_locs_centered[..., 1:], city_locs_centered[..., :1]
            )
            cities_feats = torch.cat([cities_feats, angle_to_depot], dim=-1)

        node_embeddings = self.init_embed(cities_feats)
        out = torch.cat([depot_embedding, node_embeddings], dim=-2)

        if self.use_matnet_init:
            b, r, c = distance.shape
            row_emb = torch.zeros(b, r, self.embed_dim, device=distance.device)
            col_emb = torch.zeros(b, c, self.embed_dim, device=distance.device)
            row_emb = self.combine_row_embed(torch.cat([row_emb, out], dim=-1))
            col_emb = self.combine_col_embed(torch.cat([col_emb, out], dim=-1))
            return row_emb, col_emb, distance

        return out, out, distance

    def _embed_with_distance(self, locs, vrp_attr, distance, phase):
        node_embeddings = self.coord_expert(locs)
        row_emb, col_emb = self.distance_expert(distance, phase)

        combined_row_emb = self.gating_network_row(node_embeddings, row_emb)
        combined_col_emb = self.gating_network_col(node_embeddings, col_emb)
        vrp_attr_emb = self.init_embed(vrp_attr)

        combined_row_output = self.combine_row_embed(
            torch.cat([combined_row_emb, vrp_attr_emb], dim=-1)
        )
        combined_col_output = self.combine_col_embed(
            torch.cat([combined_col_emb, vrp_attr_emb], dim=-1)
        )
        return combined_row_output, combined_col_output, distance


class CoordinateExpert(nn.Module):
    """Processes spatial coordinate information."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.init_embed_depot = nn.Linear(2, embed_dim)
        self.init_embed = nn.Linear(input_dim, embed_dim)

    def forward(self, locs):
        depot, cities = locs[:, :1, :], locs[:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)

        city_locs_centered = cities - depot
        angle_to_depot = torch.atan2(
            city_locs_centered[..., 1:], city_locs_centered[..., :1]
        )
        cities_feats = torch.cat([cities, angle_to_depot], dim=-1)

        node_embeddings = self.init_embed(cities_feats)
        return torch.cat([depot_embedding, node_embeddings], dim=-2)


class DistanceExpert(nn.Module):
    """Processes distance matrix information."""

    def __init__(self, embed_dim, sample_type, sample_size, linear_bias=True):
        super().__init__()
        self.sample_type = sample_type
        self.sample_size = sample_size
        self.row_embed = nn.Linear(sample_size, embed_dim, linear_bias)
        self.col_embed = nn.Linear(sample_size, embed_dim, linear_bias)
        self.row_combine_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)
        self.col_combine_embed = nn.Linear(embed_dim * 2, embed_dim, linear_bias)

    def forward(self, distance_matrix, phase):
        B, N, _ = distance_matrix.shape
        random_indices_expanded = self._sample_indices(distance_matrix, phase, B, N)
        row_distance = distance_matrix.gather(
            2, random_indices_expanded
        )  # [B, N, sample_size]
        col_distance = distance_matrix.transpose(1, 2).gather(
            2, random_indices_expanded
        )  # [B, N, sample_size]
        row_emb = self.row_embed(row_distance.sort(dim=-1).values)
        col_emb = self.col_embed(col_distance.sort(dim=-1).values)
        return row_emb, col_emb

    def _sample_indices(self, distance, phase, B, N):
        if self.sample_type == "random":
            if phase == "train":
                random_indices = torch.randint(
                    0, N, (1, self.sample_size), device=distance.device
                )  # (1, sample_size)
                return random_indices.unsqueeze(1).expand(
                    B, N, self.sample_size
                )  # (B, N, sample_size)
            else:
                random_indices = torch.randint(
                    0, N, (8, 1, self.sample_size), device=distance.device
                )  # (8, 1, sample_size)
                return (
                    random_indices.unsqueeze(1)
                    .expand(8, B // 8, N, self.sample_size)
                    .reshape(8 * B // 8, N, self.sample_size)
                )  # (8B, N, sample_size)
        elif self.sample_type == "prob":
            indicies = torch.arange(N, device=distance.device)
            processed_distance = distance.clone()
            processed_distance[:, indicies, indicies] = 1e6
            inverse_distance = 1 / (processed_distance + 1e-6)  # Avoid division by zero
            probabilities = inverse_distance / inverse_distance.sum(
                dim=-1, keepdim=True
            )  # Normalize
            probabilities = probabilities.reshape(B * N, -1)
            sampled_indices = torch.multinomial(
                probabilities, self.sample_size, replacement=False
            )  # [B, N, sample_size]
            return sampled_indices.reshape(B, N, self.sample_size)


class ContextualGating(nn.Module):
    """Combines embeddings using contextual gating."""

    def __init__(self, embed_dim):
        super().__init__()
        self.gating_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, coord_feat, dist_feat):
        combined_feat = torch.cat([coord_feat, dist_feat], dim=-1)
        gating_weights = self.sigmoid(self.gating_fc(combined_feat))
        return gating_weights * coord_feat + (1 - gating_weights) * dist_feat
