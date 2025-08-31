import torch
from pathlib import Path

def save_lora_adapters(model: torch.nn.Module, path: str):
    """
    Save only LoRA adapter weights (A/B or down/up) into a dict for compact checkpointing.
    """
    sd = {}
    for n, m in model.named_modules():
        # LoRALinear
        if m.__class__.__name__ == "LoRALinear":
            sd[f"{n}.lora_A"] = m.lora_A.detach().cpu()
            sd[f"{n}.lora_B"] = m.lora_B.detach().cpu()
            sd[f"{n}.alpha"] = torch.tensor(m.alpha)
            sd[f"{n}.r"] = torch.tensor(m.r)
        # LoRAConv1x1
        if m.__class__.__name__ == "LoRAConv1x1":
            sd[f"{n}.down.weight"] = m.down.weight.detach().cpu()
            sd[f"{n}.up.weight"]   = m.up.weight.detach().cpu()
            sd[f"{n}.alpha"] = torch.tensor(m.alpha)
            sd[f"{n}.r"] = torch.tensor(m.r)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    print("Saved LoRA adapters to", path)

def load_lora_adapters(model: torch.nn.Module, path: str, strict: bool = False):
    sd = torch.load(path, map_location="cpu")
    missing = []
    for n, m in model.named_modules():
        if m.__class__.__name__ == "LoRALinear":
            a_key = f"{n}.lora_A"
            b_key = f"{n}.lora_B"
            if a_key in sd and b_key in sd:
                m.lora_A.data.copy_(sd[a_key].to(m.lora_A.device))
                m.lora_B.data.copy_(sd[b_key].to(m.lora_B.device))
            else:
                missing.append(n)
        if m.__class__.__name__ == "LoRAConv1x1":
            down_k = f"{n}.down.weight"
            up_k   = f"{n}.up.weight"
            if down_k in sd and up_k in sd:
                m.down.weight.data.copy_(sd[down_k].to(m.down.weight.device))
                m.up.weight.data.copy_(sd[up_k].to(m.up.weight.device))
            else:
                missing.append(n)
    if missing and strict:
        raise RuntimeError("Missing LoRA keys for: " + ", ".join(missing))
    if missing:
        print("load_lora_adapters: missing adapters for:", missing)


def merge_lora_into_base(model: torch.nn.Module):
    """
    For LoRALinear, merges B@A*scaling into base.weight in-place.
    For LoRAConv1x1 we skip merge (could be implemented by converting up/down convs to dense)
    """
    for n, m in model.named_modules():
        if m.__class__.__name__ == "LoRALinear":
            with torch.no_grad():
                delta = (m.lora_B @ m.lora_A) * (m.scaling)  # out x in
                # base.weight shape: (out, in)
                m.base.weight.data.add_(delta.to(m.base.weight.data.dtype))
                # after merging you could zero-out adapters or remove them
        # LoRAConv1x1 merging is more complex; keep adapters if needed
    print("Merged LoRA into base weights where possible.")
