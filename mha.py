import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionScratch(nn.Module):
    """
    Multi-Head Self/Encoder-Decoder Attention implemented from scratch using only
    matmul + reshapes (no nn.Linear layers).

    Shapes use batch-first convention: (B, T, D).

    Args:
        d_model: model (embedding) dimension.
        num_heads: number of attention heads.
        dropout_p: dropout probability on attention weights.
        bias: whether to include bias terms for the projections.
        causal: if True, applies a causal mask (no peeking ahead).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = dropout_p
        self.causal = causal

        # Projection weights (no nn.Linear) — just Parameters used with matmul
        # Input: (B, T, D) -> (B, T, D) per projection
        self.W_q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_k = nn.Parameter(torch.empty(d_model, d_model))
        self.W_v = nn.Parameter(torch.empty(d_model, d_model))
        self.b_q = nn.Parameter(torch.empty(d_model)) if bias else None
        self.b_k = nn.Parameter(torch.empty(d_model)) if bias else None
        self.b_v = nn.Parameter(torch.empty(d_model)) if bias else None

        # Output projection: concat heads (B, T, H*Dh=D) -> (B, T, D)
        self.W_o = nn.Parameter(torch.empty(d_model, d_model))
        self.b_o = nn.Parameter(torch.empty(d_model)) if bias else None

        self.attn_dropout = nn.Dropout(dropout_p)
        self.out_dropout = nn.Dropout(dropout_p)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init similar to nn.Linear defaults
        for W in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(W)
        for b in (self.b_q, self.b_k, self.b_v, self.b_o):
            if b is not None:
                nn.init.zeros_(b)

    @staticmethod
    def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        (B, T, D) -> (B, H, T, Dh)
        """
        B, T, D = x.shape
        Dh = D // num_heads
        x = x.view(B, T, num_heads, Dh)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, T, Dh) -> (B, T, H*Dh)
        """
        B, H, T, Dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H * Dh)

    def _project(self, x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Affine projection using matmul only.
        x: (B, T, D), W: (D, D), b: (D,) or None -> (B, T, D)
        """
        # (B, T, D) @ (D, D) -> (B, T, D)
        y = torch.matmul(x, W)
        if b is not None:
            y = y + b
        return y

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        Args:
            x_q: query input (B, T_q, D)
            x_kv: key/value input (B, T_kv, D). If None, uses x_q (self-attention).
            attn_mask: additive mask broadcastable to (B, H, T_q, T_kv). Use 0 to keep, -inf to mask.
            key_padding_mask: boolean mask (B, T_kv), True for positions to mask (padding).
            need_weights: if True, also return averaged attention weights over heads (B, T_q, T_kv).
        Returns:
            y: output (B, T_q, D)
            attn_probs_avg (optional): (B, T_q, T_kv)
        """
        if x_kv is None:
            x_kv = x_q

        B, T_q, D = x_q.shape
        B2, T_kv, D2 = x_kv.shape
        assert D == self.d_model and D2 == self.d_model and B == B2

        # Projections via matmul
        Q = self._project(x_q, self.W_q, self.b_q)  # (B, T_q, D)
        K = self._project(x_kv, self.W_k, self.b_k)  # (B, T_kv, D)
        V = self._project(x_kv, self.W_v, self.b_v)  # (B, T_kv, D)

        # Split into heads
        Q = self._split_heads(Q, self.num_heads)  # (B, H, T_q, Dh)
        K = self._split_heads(K, self.num_heads)  # (B, H, T_kv, Dh)
        V = self._split_heads(V, self.num_heads)  # (B, H, T_kv, Dh)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        # (B, H, T_q, Dh) @ (B, H, Dh, T_kv) -> (B, H, T_q, T_kv)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        # Masks
        if attn_mask is not None:
            # attn_mask is expected additive; broadcast to (B, H, T_q, T_kv)
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_kv) with True for pads -> set to -inf
            mask = key_padding_mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        if self.causal:
            # Causal mask: prevent attending to future positions in keys
            # shape (T_q, T_kv)
            i = torch.arange(T_q, device=scores.device)[:, None]
            j = torch.arange(T_kv, device=scores.device)[None, :]
            causal = (j > i)
            scores = scores.masked_fill(causal, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # (B, H, T_q, T_kv) @ (B, H, T_kv, Dh) -> (B, H, T_q, Dh)
        context = torch.matmul(attn_probs, V)

        # Merge heads and output projection
        context = self._merge_heads(context)  # (B, T_q, D)
        y = torch.matmul(context, self.W_o)  # (B, T_q, D)
        if self.b_o is not None:
            y = y + self.b_o
        y = self.out_dropout(y)

        if need_weights:
            # average over heads
            attn_probs_avg = attn_probs.mean(dim=1)
            return y, attn_probs_avg
        return y


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H = 2, 5, 32, 4
    x = torch.randn(B, T, D)
    mha = MultiHeadAttentionScratch(d_model=D, num_heads=H, dropout_p=0.1, causal=False)

    # Self-attention
    y, w = mha(x, need_weights=True)
    print("y:", y.shape)        # (B, T, D)
    print("attn:", w.shape)     # (B, T, T)

    # Cross-attention example (encoder-decoder style)
    enc = torch.randn(B, 7, D)
    y2 = mha(x_q=x, x_kv=enc)
    print("y2:", y2.shape)      # (B, T, D)
