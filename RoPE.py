import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):

    """
    Standard RoPE implementation.
    Rotates the Query and Key vectors to encode relative position.
    """
    
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x):

        # x shape: (Batch, Heads, Time, Head_Dim)
        B, H, T, D = x.shape
        
        # Create position indices [0, 1, ..., T-1]
        t = torch.arange(T, device=x.device).type_as(self.inv_freq)
        
        # Outer product to get frequencies for each position
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1) # (T, D)
        
        # Expand for broadcasting: (1, 1, T, D)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


    def apply_rotary_emb(self, x, cos, sin):
        # Rotate every two elements: [x1, x2] -> [-x2, x1]
        # x shape: (B, H, T, D)
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        rotated_x = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated_x * sin)