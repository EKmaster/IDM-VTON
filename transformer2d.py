import math
import torch
import torch.nn as nn

def sinusoidal_2d_pe(H, W, C, device):
    # C must be divisible by 4
    assert C % 4 == 0, "C must % 4 == 0 for this PE"
    c = C // 4
    y = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).float()  # H x W
    x = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).float()  # H x W
    omega = torch.exp(torch.arange(c, device=device).float() * -(math.log(10000.0) / c))
    pe_y = torch.einsum("hw,c->hwc", y, omega)
    pe_x = torch.einsum("hw,c->hwc", x, omega)
    pe = torch.cat([pe_y.sin(), pe_y.cos(), pe_x.sin(), pe_x.cos()], dim=-1)  # H x W x C
    return pe.unsqueeze(0)  # 1 x H x W x C

class Transformer2D(nn.Module):
    def __init__(self, dim, n_heads=None, mlp_ratio=4.0, dropout=0.0, cross_attn=False, use_pe=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads or max(1, min(8, dim // 64))
        self.cross_attn = cross_attn
        self.use_pe = use_pe

        self.ln1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.attn = nn.MultiheadAttention(dim, self.n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_hidden_states=None):
        # x: (B, C, H, W) -> to (B, HW, C)
        B, C, H, W = x.shape
        t = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # B, S, C
        if self.use_pe:
            pe = sinusoidal_2d_pe(H, W, C, device=t.device)  # 1,H,W,C
            t = t + pe.reshape(1, H * W, C)

        h = self.ln1(t)
        q = self.q(h)
        kv = self.kv(h)
        k, v = kv.chunk(2, dim=-1)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        t = t + self.dropout(attn_out)
        t = t + self.mlp(self.ln2(t))
        out = t.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out
