"""Lazy, model-specific CUDA/MPS/CPU dispatch; importing this module never imports torch."""


class DeviceUnavailableError(RuntimeError):
    pass


def probe_device(device, family):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        return False
    if device == "mps":
        if not torch.backends.mps.is_available():
            return False
        if family == "tabicl":
            from tabicl._torch_devices import mps_possibly_faulty
            if mps_possibly_faulty():
                return False
    try:
        result = torch.eye(2, device=device) @ torch.ones(2, 1, device=device)
        return float(result.cpu().sum()) == 2.0
    except RuntimeError:
        return False


def select_device(requested, family, *, probe=None):
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto/cuda/mps/cpu")
    if family == "lightgbm":
        if requested not in {"auto", "cpu"}:
            raise DeviceUnavailableError("This LightGBM experiment requires CPU")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    probe = probe or probe_device
    for device in (("cuda", "mps", "cpu") if requested == "auto" else (requested,)):
        if probe(device, family):
            return device
    raise DeviceUnavailableError(f"Requested device is unavailable: {requested}")
