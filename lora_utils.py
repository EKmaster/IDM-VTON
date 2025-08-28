# lora_utils.py
import torch
import torch.nn as nn
from typing import Iterable
from lora import LoRALinear, LoRAConv1x1

def is_target_name(name: str, keywords: Iterable[str]):
    return any(k in name for k in keywords)

def wrap_unet_with_lora(model: nn.Module,
                        target_keywords=("to_q", "to_k", "to_v", "to_out", "proj_out"),
                        r:int = 4,
                        alpha:float = 16.0,
                        dropout:float = 0.0):
    """
    Recursively replaces Linear and 1x1 Conv layers whose attribute name contains any
    of target_keywords with LoRA wrappers.
    Returns list of (module_path, orig_module) for possible restoration and a list of new trainable params.
    """
    replaced = []
    trainable_params = []

    # We need parent references to set attributes; do a recursive traversal:
    def recursion(parent):
        for child_name, child in list(parent.named_children()):
            # Recurse into child first
            recursion(child)

            # Decide whether to wrap this attribute
            attr_name = child_name  # attribute name within parent
            full_name = f"{parent.__class__.__name__}.{attr_name}"

            if isinstance(child, nn.Linear) and is_target_name(attr_name, target_keywords):
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, wrapped)
                replaced.append((full_name, child))
                trainable_params += [wrapped.lora_A, wrapped.lora_B]

            # For Conv2d 1x1 project layers (some implementations use 1x1 convs)
            elif isinstance(child, nn.Conv2d) and child.kernel_size == (1,1) and is_target_name(attr_name, target_keywords):
                wrapped = LoRAConv1x1(child, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, wrapped)
                replaced.append((full_name, child))
                trainable_params += list(wrapped.down.parameters()) + list(wrapped.up.parameters())

    recursion(model)
    return replaced, trainable_params
