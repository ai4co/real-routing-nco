from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"
        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


class DistAngleFusion(nn.Module):
    def __init__(self, embed_dim: int, use_duration_matrix: bool = False):
        """
        embed_dim: the embedding dimension for row_emb and col_emb
        hidden_dim: the intermediate dimension used inside the MLP (can be adjusted as desired)
        """
        super().__init__()
        self.embed_dim = embed_dim
        if use_duration_matrix:
            self.dur_emb = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),  # output shape: (B, R, C, E)
            )
            self.combined_emb = nn.Linear(2 * embed_dim, embed_dim)
        # coordinate -> coord_emb
        # Concatenate row_emb and col_emb, then (2E) -> (hidden_dim) -> (embed_dim)
        self.dist_emb = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),  # output shape: (B, R, C, E)
        )
        self.angle_emb = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),  # output shape: (B, R, C, E)
        )

        # Gate: concatenate dist_emb and coord_emb => (2E) -> scalar gate \in (0,1)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())
        # Projection for final adapt_bias: (E) -> (1)
        self.out_lin = nn.Linear(embed_dim, 1)

    def forward(
        self, coords: torch.Tensor, cost_mat: torch.Tensor, duration_mat: torch.Tensor
    ):
        """
        coords: shape (B, N, 2)
        cost_mat: shape (B, N, N)
        """
        B, N, _ = cost_mat.shape

        # coords: (batch_size, N, 2), where N is the total number of nodes (depot + cities)
        batch_size, N, _ = coords.shape

        # Calculate the pairwise differences
        coords_expanded_1 = coords.unsqueeze(2)  # (batch_size, N, 1, 2)
        coords_expanded_2 = coords.unsqueeze(1)  # (batch_size, 1, N, 2)
        pairwise_diff = coords_expanded_1 - coords_expanded_2  # (batch_size, N, N, 2)

        # Compute pairwise angles using atan2
        angles = torch.atan2(
            pairwise_diff[..., 1], pairwise_diff[..., 0]
        )  # (batch_size, N, N)
        dist_emb = self.dist_emb(cost_mat.unsqueeze(-1))  # shape (B, N, N, E)
        angle_emb = self.angle_emb(angles.unsqueeze(-1))  # shape (B, N, N, E)
        if duration_mat is not None:
            dur_emb = self.dur_emb(duration_mat.unsqueeze(-1))
        # 5) Calculate the gate
        gate_in = torch.cat([dist_emb, angle_emb], dim=-1)  # shape (B, N, N, 2E)
        g = self.gate(gate_in)  # shape (B, N, N, 1) in (0, 1)

        # 6) Weighted sum of dist_emb vs coord_emb
        if duration_mat is not None:
            fused_emb = self.combined_emb(
                torch.cat([g * dur_emb + (1 - g) * angle_emb, dur_emb], dim=-1)
            )
        else:
            fused_emb = g * dist_emb + (1 - g) * angle_emb  # shape (B, N, N, E)

        # 7) Generate the adapt_bias (scalar) for AFTFull
        #    (B, N, N, E) -> linear -> (B, N, N, 1) -> squeeze(-1) -> (B, N, N)

        adapt_bias = self.out_lin(fused_emb).squeeze(-1)  # shape (B, N, N)

        return adapt_bias


class AFTFull(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        """
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x, y=None, adapt_bias=None):
        B, T, _ = x.shape
        if y is None:
            y = x
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(y).view(B, T, self.hidden_dim)
        V = self.to_v(y).view(B, T, self.hidden_dim)
        Q_sig = torch.sigmoid(Q)

        adapt_bias = torch.softmax(adapt_bias, dim=-1)
        K = torch.softmax(K, dim=1)
        temp = torch.exp(adapt_bias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(adapt_bias) @ torch.exp(K))

        Yt = torch.mul(Q_sig, weighted)
        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt


class TransformerFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(**parallel_gated_kwargs)
        else:
            ffn = FeedForward(embed_dim=embed_dim, feedforward_hidden=feedforward_hidden)

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(embed_dim=embed_dim, normalization=normalization),
                "ffn": ffn,
                "norm2": Normalization(embed_dim=embed_dim, normalization=normalization),
            }
        )

    def forward(self, x, x_old):
        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x


class AttnFree_Block(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = nn.Parameter(torch.ones(1))
        self.attn_free = AFTFull(dim=embed_dim, hidden_dim=embed_dim)
        self.multi_head_combine = nn.Linear(embed_dim, embed_dim)
        self.angle_distance_fusion = DistAngleFusion(
            embed_dim=embed_dim,
            use_duration_matrix=kwargs.get("use_duration_matrix", False),
        )
        self.feed_forward = TransformerFFN(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
        )

        self.norm1 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm2 = Normalization(embed_dim=embed_dim, normalization=normalization)
        self.norm3 = Normalization(embed_dim=embed_dim, normalization=normalization)

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_mat=None):
        # q shape: (batch, row_cnt, self.embed_dim)
        # k,v shape: (batch, col_cnt, self.embed_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb = self.norm1(row_emb)
        col_emb = self.norm2(col_emb)

        # Nerual Adaptive Bias (NAB)
        adapt_bias = (
            self.angle_distance_fusion(coords, cost_mat, duration_mat) * self.alpha
        )
        out_concat = self.attn_free(row_emb, y=col_emb, adapt_bias=adapt_bias)

        multi_head_out = self.multi_head_combine(out_concat)
        multi_head_out = self.norm3(multi_head_out)

        # shape: (batch, row_cnt, embedding)
        ffn_out = self.feed_forward(multi_head_out, row_emb)

        return ffn_out


class Attn_Free_Layer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.row_encoding_block = AttnFree_Block(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **kwargs,
        )
        self.col_encoding_block = AttnFree_Block(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **kwargs,
        )

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_mat=None):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(
            row_emb, col_emb, cost_mat, coords, duration_mat
        )
        if duration_mat is not None:
            trans_duration_mat = duration_mat.transpose(1, 2)
        else:
            trans_duration_mat = None
        col_emb_out = self.col_encoding_block(
            col_emb, row_emb, cost_mat.transpose(1, 2), coords, trans_duration_mat
        )

        return row_emb_out, col_emb_out


class AttnFreeNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        feedforward_hidden: int = 512,
        num_layers: int = 3,
        normalization: Optional[str] = "instance",
        parallel_gated_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Attn_Free_Layer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, row_emb, col_emb, cost_mat, coords, duration_mat=None):
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat, coords, duration_mat)

        return row_emb, col_emb


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.W1 = nn.Linear(embed_dim, feedforward_hidden)
        self.W2 = nn.Linear(feedforward_hidden, embed_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))
