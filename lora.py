import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# LoRA for Linear layers
# ----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 16.0, dropout: float = 0.0):
        """
        This function wraps an existing nn.Linear layer with LoRA adapters.
        base: the original Linear layer (will be frozen)
        r: LoRA rank
        alpha: scaling factor (alpha / r applied)
        dropout: optional dropout on adapter input
        """
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # LoRA params: A: in->r, B: r->out (we use small dense layers)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

        # Initialization: A random, B zero so initial behavior == base
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base layer parameters
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # base result
        y = self.base(x)
        # LoRA path: x -> A.T -> B.T -> scaling
        # x shape: (..., in_features)
        # compute x @ A.T -> (..., r)
        x_drop = self.dropout(x)
        lora_down = torch.matmul(x_drop, self.lora_A.t())    # (..., r)
        lora_up   = torch.matmul(lora_down, self.lora_B.t()) # (..., out_features)
        return y + lora_up * self.scaling

    def extra_repr(self):
        return f"LoRALinear(r={self.r}, alpha={self.alpha})"



class LoRAConv1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int = 4, alpha: float = 16.0, dropout: float = 0.0):
        """
        Wraps an existing nn.Conv2d with kernel_size==1.
        For attention projection convs, 1x1 LoRA is effective and efficient.
        """
        super().__init__()
        assert isinstance(base, nn.Conv2d) and base.kernel_size == (1, 1)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # Down and Up 1x1 convs
        self.down = nn.Conv2d(base.in_channels, r, kernel_size=1, bias=False)
        self.up   = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)

        # init: down small, up zero so no initial change
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        y = self.base(x)
        z = self.down(self.dropout(x))
        z = self.up(z) * self.scaling
        return y + z

    def extra_repr(self):
        return f"LoRAConv1x1(r={self.r}, alpha={self.alpha})"
