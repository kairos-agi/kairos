"""Checkpoint helpers for Kairos training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def _trainable_parameter_names(model: torch.nn.Module) -> set[str]:
    names = getattr(model, "trainable_param_names", None)
    if callable(names):
        return set(names())
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def trainable_state_dict(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    """Return all trainable parameters using their complete model keys."""

    state_dict = model.state_dict()
    export = getattr(model, "export_trainable_state_dict", None)
    if callable(export):
        state_dict = export(state_dict)
    else:
        names = _trainable_parameter_names(model)
        state_dict = {
            name: value for name, value in state_dict.items() if name in names
        }
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in state_dict.items()
    }


def save_trainable_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save the model's trainable parameters to a safetensors file."""

    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_metadata = None
    if metadata:
        string_metadata = {str(key): str(value) for key, value in metadata.items()}
    save_file(
        dict(trainable_state_dict(model)),
        str(path),
        metadata=string_metadata,
    )


def load_trainable_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    strict: bool = False,
) -> Any:
    """Load matching parameters from a trainable-parameter checkpoint.

    Partial checkpoints are accepted by default.  With ``strict=True``, every
    parameter currently marked trainable must be present and the checkpoint
    must not contain keys unknown to the model.
    """

    from safetensors.torch import load_file

    state_dict = load_file(str(path), device="cpu")
    trainable_names = _trainable_parameter_names(model)
    load_state_dict = {
        name: value
        for name, value in state_dict.items()
        if name in trainable_names
    }
    result = model.load_state_dict(load_state_dict, strict=False)
    if strict:
        state_keys = set(state_dict)
        missing_keys = sorted(trainable_names - state_keys)
        unexpected_keys = sorted(state_keys - trainable_names)
        if missing_keys or unexpected_keys:
            details = []
            if missing_keys:
                details.append(f"missing trainable keys: {missing_keys}")
            if unexpected_keys:
                details.append(f"unexpected keys: {unexpected_keys}")
            raise RuntimeError("; ".join(details))
    return result


def save_accelerate_state(accelerator: Any, output_dir: str | Path) -> None:
    """Save optimizer, scheduler, and engine state for resume."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(output_dir))


__all__ = [
    "load_trainable_checkpoint",
    "save_accelerate_state",
    "save_trainable_checkpoint",
    "trainable_state_dict",
]
