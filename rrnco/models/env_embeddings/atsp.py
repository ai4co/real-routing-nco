import torch
import torch.nn as nn


class ATSPInitEmbedding(nn.Module):
    """Initial embedding for the Asymmetric TSP (ATSP).
    Embeds the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - distance: distance between the nodes
    """

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        use_coords=True,
        use_polar_feats=False,
        use_dist=True,
        use_matnet_init=True,
        sample_type="prob",
        sample_size=25,
    ):
        super(ATSPInitEmbedding, self).__init__()
        self.use_coords = use_coords
        self.use_dist = use_dist
        self.sample_type = sample_type
        self.sample_size = sample_size

        self.init_embed = nn.Linear(2, embed_dim, linear_bias) if use_coords else None
        self.row_embed = nn.Linear(sample_size, embed_dim, linear_bias)
        self.col_embed = nn.Linear(sample_size, embed_dim, linear_bias)

        if use_coords and use_dist:
            self.gating_network_row = ContextualGating(embed_dim)
            self.gating_network_col = ContextualGating(embed_dim)

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

    def forward(self, td, phase):
        locs, distance = td["locs"].float(), td["distance_matrix"]
        B, N, _ = locs.shape

        if self.use_coords:
            node_embeddings = self.init_embed(locs)  # [B, N, embed_dim]

            if self.use_dist:
                random_indices_expanded = self._sample_indices(distance, phase, B, N)
                row_distance = distance.gather(
                    2, random_indices_expanded
                )  # [B, N, sample_size]
                col_distance = distance.transpose(1, 2).gather(
                    2, random_indices_expanded
                )  # [B, N, sample_size]

                row_emb = self.row_embed(row_distance.sort(dim=-1).values)
                col_emb = self.col_embed(col_distance.sort(dim=-1).values)

                combined_row_emb = self.gating_network_row(node_embeddings, row_emb)
                combined_col_emb = self.gating_network_col(node_embeddings, col_emb)

                return combined_row_emb, combined_col_emb, distance

            return node_embeddings.clone(), node_embeddings.clone(), distance

        # When use_coords is False
        random_indices_expanded = self._sample_indices(distance, phase, B, N)
        row_distance = distance.gather(2, random_indices_expanded)  # [B, N, sample_size]
        col_distance = distance.transpose(1, 2).gather(
            2, random_indices_expanded
        )  # [B, N, sample_size]

        row_emb = self.row_embed(row_distance)
        col_emb = self.col_embed(col_distance)

        return row_emb, col_emb, distance


class ContextualGating(nn.Module):
    def __init__(self, embed_dim):
        super(ContextualGating, self).__init__()
        self.gating_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, coord_feat, dist_feat):
        combined_feat = torch.cat([coord_feat, dist_feat], dim=-1)  # [B, N, 2*embed_dim]
        gating_weights = self.sigmoid(self.gating_fc(combined_feat))  # [B, N, embed_dim]
        return gating_weights * coord_feat + (1 - gating_weights) * dist_feat
