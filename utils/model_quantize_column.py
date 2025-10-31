import torch
import torch.nn as nn


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 2048,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 16,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}

LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,  # Vocabulary size
    "context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 3072,  # Embedding dimension
    "n_heads": 24,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
    "rope_freq": {  # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}


class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

        # Reusuable utilities
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


def quantize_row_matrix_int8_symmetric(mat:torch.Tensor):
    """
    Symmetric quantization to int8 on a per-row basis.
    mat: input float tensor (e.g., torch.float32 or torch.bfloat16)
    """
    N, M = mat.shape
    qmin = -128
    qmax = 127
    
    max_vals, _ = torch.max(torch.abs(mat), dim=1, keepdim=True)  # shape (N, 1)
    scales = (max_vals / qmax).squeeze(1)  # shape (N,)
    
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(1)), qmin, qmax).to(torch.int8)  # shape (N, M)
    
    scales = scales.to(torch.float32)
    return q_mat, scales


def quantized_column_matrix_int_symmetric(mat:torch.Tensor):
    """
    Symmetric quantization to int8 on a per-column basis.
    mat: input float tensor (e.g., torch.float32 or torch.bfloat16)
    """
    N, M = mat.shape
    qmin = -128
    qmax = 127
    
    max_vals, _ = torch.max(torch.abs(mat), dim=0, keepdim=True)  # shape (1, M)
    scales = (max_vals / qmax).squeeze(0)  # shape (M,)
    
    q_mat = torch.clamp(torch.round(mat / scales.unsqueeze(0)), qmin, qmax).to(torch.int8)  # shape (N, M)
    
    scales = scales.clone().detach().to(torch.float32)
    return q_mat, scales


def de_quantize_row_matrix_int8_symmetric_matmul(q_mat:torch.Tensor, x_scale:torch.Tensor, w_scale: torch.Tensor,\
                                        out_dtype=torch.float16):
    """
    Dequantize int8 matrix to float on a per-row basis.
    q_mat: input int8 tensor (shape (N, M))
    scales: scale factors for each row (shape (N,))
    """
    output = q_mat.to(torch.float32) 
    output = output * x_scale[:, None] * w_scale[None, :]
    output = output.to(out_dtype)
    return output


# TODO: Replace with actual int8 matmul implementation
def dummy_int8_matmul(A_int8: torch.Tensor, B_int: torch.Tensor, out_dtype=torch.int32):
    """
    This is a dummy int8 matrix multiplication function.
    """
    if A_int8.dtype != torch.int8 or B_int.dtype != torch.int8:
        raise ValueError("Both A and B must be int8 tensors.")
    result_float = torch.matmul(A_int8.float(), B_int.float())
    return result_float.to(out_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Parameter(torch.empty((d_in, num_kv_groups * self.head_dim), dtype=dtype))
        self.W_value = nn.Parameter(torch.empty((d_in, num_kv_groups * self.head_dim), dtype=dtype))
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Parameter(torch.empty((d_in,d_out), dtype=dtype))
        self.out_proj = nn.Parameter(torch.empty((d_out, d_out), dtype=dtype))

        # Quantization parameters
        self.register_buffer("W_query_scale", torch.ones(d_out, dtype=torch.float32), persistent=False)
        self.register_buffer("W_key_scale", torch.ones(num_kv_groups * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("W_value_scale", torch.ones(num_kv_groups * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("out_proj_scale", torch.ones(d_out, dtype=torch.float32), persistent=False)

        self.register_buffer("W_query_q",torch.empty_like(self.W_query, dtype=torch.int8),persistent=False)
        self.register_buffer("W_key_q", torch.empty_like(self.W_key, dtype=torch.int8), persistent=False)
        self.register_buffer("W_value_q", torch.empty_like(self.W_value, dtype=torch.int8),persistent=False)
        self.register_buffer("out_proj_q", torch.empty_like(self.out_proj, dtype=torch.int8),persistent=False)

        # Calibration flag
        self.register_buffer("x_min", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("x_max", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("x_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("x_o_min", torch.tensor(0.0, dtype=torch.float32), persistent=False)  # output of scaled dot-product
        self.register_buffer("x_o_max", torch.tensor(0.0, dtype=torch.float32), persistent=False)  # output of scaled dot-product
        self.register_buffer("x_o_scale", torch.tensor(1.0, dtype=torch.float32), persistent=False)  # output of scaled dot-product
        self.calibrating = True

        self.is_quantized = False

    @torch.no_grad()
    def observe_activation(self, x):
        min_val = x.min()
        max_val = x.max()
        self.x_min = min(self.x_min, min_val)
        self.x_max = max(self.x_max, max_val)
        
    @torch.no_grad()
    def observe_output_activation(self, x_o):
        min_val_o = x_o.min()
        max_val_o = x_o.max()
        self.x_o_min = min(self.x_o_min, min_val_o)
        self.x_o_max = max(self.x_o_max, max_val_o)

    @torch.no_grad()
    def quantize_weights(self):
        # Quantize W_query
        W_query_q, W_query_scale = quantized_column_matrix_int_symmetric(self.W_query)
        self.W_query_q.copy_(W_query_q)
        self.W_query_scale.copy_(W_query_scale)

        # Quantize W_key
        W_key_q, W_key_scale = quantized_column_matrix_int_symmetric(self.W_key)
        self.W_key_q.copy_(W_key_q)
        self.W_key_scale.copy_(W_key_scale)

        # Quantize W_value
        W_value_q, W_value_scale = quantized_column_matrix_int_symmetric(self.W_value)
        self.W_value_q.copy_(W_value_q)
        self.W_value_scale.copy_(W_value_scale)

        # Quantize out_proj
        out_proj_q, out_proj_scale = quantized_column_matrix_int_symmetric(self.out_proj)
        self.out_proj_q.copy_(out_proj_q)
        self.out_proj_scale.copy_(out_proj_scale)

        # Free up memory
        del self.W_query
        del self.W_key
        del self.W_value
        del self.out_proj

        self.is_quantized = True
        print("[INFO] Quantized weights to int8. Deleted original weights to save memory.")

        # Compute activation quantization parameters per row (for input x)
        x_max = max(abs(self.x_min), abs(self.x_max))
        x_scale = x_max / 127.0
        self.x_scale.copy_(x_scale)
        print(f"[INFO] Activation quantization scale set to {x_scale.item():.6f}")
        
        # Compute output activation quantization parameters
        x_o_max = max(abs(self.x_o_min), abs(self.x_o_max))
        x_o_scale = x_o_max / 127.0
        self.x_o_scale.copy_(x_o_scale)
        print(f"[INFO] Output activation quantization scale set to {x_o_scale.item():.6f}")
        

    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_in = x.shape

        # 1. Query projection
        if (self.is_quantized == False) and (self.calibrating == False):
            queries = x @ self.W_query  # Shape: (b, num_tokens, d_out)
        elif (self.is_quantized == False) and (self.calibrating == True):  # Calibration
            self.observe_activation(x)
            queries = x @ self.W_query  # Shape: (b, num_tokens, d_out)
        else:
            # Quantize input
            if x.dtype != torch.int8:
                X_q = torch.clamp(torch.round(x / self.x_scale), -128, 127).to(torch.int8)
            else:
                X_q = x
            queries_quant = dummy_int8_matmul(X_q, self.W_query_q, out_dtype=torch.int32)
            queries = queries_quant * self.W_query_scale[None, :] * self.x_scale
            queries = queries.to(x.dtype)

        # 2. Key projections
        if self.is_quantized == False:
            keys = x @ self.W_key  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        elif (self.is_quantized == False) and (self.calibrating == True):  # Calibration
            self.observe_activation(x)
            keys = x @ self.W_key  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        else:
            if x.dtype != torch.int8:
                X_q = torch.clamp(torch.round(x / self.x_scale), -128, 127).to(torch.int8)
            else:
                X_q = x
            keys_quant = dummy_int8_matmul(X_q, self.W_key_q, out_dtype=torch.int32)
            keys = keys_quant * self.W_key_scale[None, :] * self.x_scale
            keys = keys.to(x.dtype)

        # 3. Value projections
        if self.is_quantized == False:
            values = (x @ self.W_value)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        elif (self.is_quantized == False) and (self.calibrating == True):  # Calibration
            self.observe_activation(x)
            values = (
                x @ self.W_value
            )  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        else:
            if x.dtype != torch.int8:
                X_q = torch.clamp(torch.round(x / self.x_scale), -128, 127).to(torch.int8)
            else:
                X_q = x
            values_quant = dummy_int8_matmul(X_q, self.W_value_q, out_dtype=torch.int32)
            values = values_quant * self.W_value_scale[None, :] * self.x_scale
            values = values.to(x.dtype)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # Expand keys and values to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)

        # Scaled dot-product attention with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        if (self.is_quantized == False) and (self.calibrating == False):
            context_vec = context_vec @ self.out_proj
        elif (self.is_quantized == False) and (self.calibrating == True):  # Calibration
            self.observe_output_activation(context_vec)
            context_vec = context_vec @ self.out_proj
        else:
            if context_vec.dtype != torch.int8:
                context_vec_q = torch.clamp(torch.round(context_vec / self.x_o_scale), -128, 127).to(torch.int8)
            else:
                context_vec_q = context_vec
                
            context_quant = dummy_int8_matmul(context_vec_q, self.out_proj_q, out_dtype=torch.int32)
            context_vec = context_quant * self.out_proj_scale[None, :] * self.x_o_scale
            context_vec = context_vec.to(x.dtype)

        return context_vec


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    freq_config=None,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):

    is_end_of_sequence = False  # Flag to indicate if EOS token is generated

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            is_end_of_sequence = True
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    if is_end_of_sequence == False:
        print(
            f"\n [WARNING]: Reached limit of {max_new_tokens} token without generating an EOS token. \n"
        )

    return idx
