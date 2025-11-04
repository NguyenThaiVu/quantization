
import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

# TODO: Replace with actual int8 matmul implementation
def dummy_int8_matmul(A_int8: torch.Tensor, B_int: torch.Tensor, out_dtype=torch.int32):
    """
    This is a dummy int8 matrix multiplication function.
    """
    if A_int8.dtype != torch.int8 or B_int.dtype != torch.int8:
        raise ValueError("Both A and B must be int8 tensors.")
    result_float = torch.matmul(A_int8.float(), B_int.float())
    return result_float.to(out_dtype)

def quantized_column_matrix_int_symmetric(mat:torch.Tensor):
    """
    Symmetric quantization to int8 on a per-column basis.
    mat: input float tensor (e.g., torch.float32 or torch.bfloat16)
    """
    qmin = -128
    qmax = 127
    
    max_vals, _ = torch.max(torch.abs(mat), dim=0, keepdim=True)  # shape (1, M)
    scales = (max_vals / qmax).squeeze(0)  # shape (M,)
    
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(0)), qmin, qmax).to(torch.int8)  # shape (N, M)
    
    scales = scales.clone().detach().to(torch.float32)
    return q_mat, scales

def quantize_row_matrix_int8_symmetric_batched(mat: torch.Tensor):
    """
    Symmetric per-row quantization for batched 3D tensor.
    mat: [B, N, D]  (float tensor)
    
    Returns:
        q_mat:   [B, N, D] int8
        scales:  [B, N]    float32  (scale per row within each batch)
    """
    qmin, qmax = -128, 127

    # Compute max abs per row (per batch) - Result shape: [B, N, 1]
    max_vals, _ = torch.max(torch.abs(mat), dim=2, keepdim=True)

    # Compute scales per row
    scales = (max_vals / qmax).clamp(min=1e-12)  # avoid div-by-zero, shape [B, N, 1]

    # Quantize
    q_mat = torch.clamp(torch.round(mat / scales), qmin, qmax).to(torch.int8)

    # Return float scales of shape [B, N]
    scales = scales.squeeze(2).to(torch.float32)
    return q_mat, scales 

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        # self.proj_q = nn.Linear(dim, dim)
        # self.proj_k = nn.Linear(dim, dim)
        # self.proj_v = nn.Linear(dim, dim)
        
        self.proj_q = nn.Parameter(torch.zeros(dim, dim))
        self.proj_q_bias = nn.Parameter(torch.zeros(dim))
        self.proj_k = nn.Parameter(torch.zeros(dim, dim))
        self.proj_k_bias = nn.Parameter(torch.zeros(dim))
        self.proj_v = nn.Parameter(torch.zeros(dim, dim))
        self.proj_v_bias = nn.Parameter(torch.zeros(dim))
        
        self.is_merge_w_and_b = False
        self.register_buffer("W_q", torch.empty(dim + 1, dim), persistent=False)
        self.register_buffer("W_k", torch.empty(dim + 1, dim), persistent=False)
        self.register_buffer("W_v", torch.empty(dim + 1, dim), persistent=False)
        
        # Quantization buffers
        # Quantization parameters
        self.register_buffer("W_q_q", torch.empty_like(self.W_q, dtype=torch.int8), persistent=False)
        self.register_buffer("W_k_q", torch.empty_like(self.W_k, dtype=torch.int8), persistent=False)
        self.register_buffer("W_v_q", torch.empty_like(self.W_v, dtype=torch.int8), persistent=False)
        
        self.register_buffer("W_q_scale", torch.empty(dim), persistent=False)
        self.register_buffer("W_k_scale", torch.empty(dim), persistent=False)
        self.register_buffer("W_v_scale", torch.empty(dim), persistent=False)
        
        self.is_quantized = False
        
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization
        
    @torch.no_grad()
    def merge_weight_bias(self):        
        W_q = torch.concat([self.proj_q, self.proj_q_bias.unsqueeze(0)], dim=0)
        W_k = torch.concat([self.proj_k, self.proj_k_bias.unsqueeze(0)], dim=0)
        W_v = torch.concat([self.proj_v, self.proj_v_bias.unsqueeze(0)], dim=0)

        self.W_q.copy_(W_q)
        self.W_k.copy_(W_k)
        self.W_v.copy_(W_v)
        
        self.is_merge_w_and_b = True
        
    @torch.no_grad()
    def quantize_weights(self):
        if self.is_merge_w_and_b == False:
            self.merge_weight_bias()
        
        W_q_q, W_q_scale = quantized_column_matrix_int_symmetric(self.W_q)
        W_k_q, W_k_scale = quantized_column_matrix_int_symmetric(self.W_k)
        W_v_q, W_v_scale = quantized_column_matrix_int_symmetric(self.W_v)
        
        self.W_q_q.copy_(W_q_q)
        self.W_k_q.copy_(W_k_q)
        self.W_v_q.copy_(W_v_q)
        
        self.W_q_scale.copy_(W_q_scale)
        self.W_k_scale.copy_(W_k_scale)
        self.W_v_scale.copy_(W_v_scale)
        
        self.is_quantized = True

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # Overall: (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        
        # Project inputs to multi-head Q, K, V
        if self.is_merge_w_and_b:
            # Append 1s to input x for bias
            B, S, D = x.size()
            x = torch.cat([x, torch.ones(B, S, 1, device=x.device, dtype=x.dtype)], dim=2)  # shape (B, S, D+1)
            if self.is_quantized:
                x_q, x_scale = quantize_row_matrix_int8_symmetric_batched(x)
                
                queries_int = dummy_int8_matmul(x_q, self.W_q_q)
                q = x_scale.unsqueeze(-1) * queries_int * self.W_q_scale[None, :]
                q = q.to(x.dtype)
                
                keys_int = dummy_int8_matmul(x_q, self.W_k_q)
                k = x_scale.unsqueeze(-1) * keys_int * self.W_k_scale[None, :]
                k = k.to(x.dtype)
                
                values_int = dummy_int8_matmul(x_q, self.W_v_q)
                v = x_scale.unsqueeze(-1) * values_int * self.W_v_scale[None, :]
                v = v.to(x.dtype)
            else:
                q = x @ self.W_q  # shape (B, S, D)
                k = x @ self.W_k  # shape (B, S, D)
                v = x @ self.W_v  # shape (B, S, D)
    
        else:
            q = x @ self.proj_q + self.proj_q_bias
            k = x @ self.proj_k + self.proj_k_bias
            v = x @ self.proj_v + self.proj_v_bias
            
        
        # Split heads (B, S, D) -> (B, H, S, W)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
