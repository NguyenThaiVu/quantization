import os 
from typing import List, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt

import torch

def compute_quantization_param(x: torch.Tensor, num_bits=8) -> Union[float, int]:
    """
    Compute the scale and zero-point for quantization.
    """
    min_val, max_val = x.min(), x.max()

    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point
    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x: torch.Tensor, scale: float, zero_point: int, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    return q_x.to(torch.int8)

def dequantize_tensor(q_x: torch.Tensor, scale: float, zero_point: int,\
                      output_dtype=torch.float32):
    de_quantized = scale * (q_x.float() - zero_point)
    return de_quantized.to(output_dtype)