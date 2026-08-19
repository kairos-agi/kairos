"""Small Accelerate-based trainer for Kairos training modules."""

from __future__ import annotations

import os
import json
import re
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .checkpointing import save_accelerate_state, save_trainable_checkpoint


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


class KairosTrainer:
    """Train the parameters exposed by a Kairos training model.

    The class has no dataset-specific logic. It expects a model
    whose ``forward`` returns either ``loss`` or ``(output_dict, loss)`` and a
    DataLoader yielding a mapping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable[Mapping[str, Any]],
        *,
        output_dir: str | os.PathLike[str],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 500,
        max_steps: int = 1000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
        save_steps: int = 1000,
        log_steps: int = 10,
        accelerator: Any | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        if hasattr(dataloader, "__len__") and len(dataloader) == 0:
            raise ValueError(
                "training dataloader is empty; check batch_size and drop_last"
            )
        self.output_dir = Path(output_dir)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.betas = betas
        self.warmup_steps = max(int(warmup_steps), 0)
        self.max_steps = int(max_steps)
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.max_grad_norm = max_grad_norm
        self.save_steps = max(int(save_steps), 1)
        self.log_steps = max(int(log_steps), 1)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        if accelerator is None:
            from accelerate import Accelerator
            from accelerate.utils import DistributedDataParallelKwargs

            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
            accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                kwargs_handlers=[ddp_kwargs],
            )
        self.accelerator = accelerator
        distributed_name = getattr(
            self.accelerator.distributed_type, "name", ""
        )
        if distributed_name not in ("NO", "MULTI_GPU", "DEEPSPEED"):
            raise ValueError(
                "KairosTrainer supports single GPU, DDP, or DeepSpeed ZeRO-1"
            )
        deepspeed_plugin = getattr(
            getattr(self.accelerator, "state", None),
            "deepspeed_plugin",
            None,
        )
        if deepspeed_plugin is not None:
            ds_config = deepspeed_plugin.deepspeed_config
            zero_stage = ds_config.get("zero_optimization", {}).get("stage")
            if int(zero_stage) != 1:
                raise ValueError("only DeepSpeed ZeRO stage 1 is supported")
            ds_accum = ds_config.get("gradient_accumulation_steps", "auto")
            if ds_accum != "auto" and int(ds_accum) != self.gradient_accumulation_steps:
                raise ValueError(
                    "DeepSpeed gradient_accumulation_steps must match trainer_cfg"
                )
            ds_clip = ds_config.get("gradient_clipping")
            expected_clip = (
                None if self.max_grad_norm is None else float(self.max_grad_norm)
            )
            clip_mismatch = (
                expected_clip is None and ds_clip not in (None, 0, 0.0)
            ) or (
                expected_clip is not None
                and (
                    ds_clip in (None, "auto")
                    or abs(float(ds_clip) - expected_clip) > 1e-12
                )
            )
            if clip_mismatch:
                raise ValueError(
                    "DeepSpeed gradient_clipping must match trainer_cfg.max_grad_norm"
                )

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("KairosTrainer received a model with no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        # When Accelerate does not split batches, AcceleratedScheduler advances
        # the underlying scheduler once per process for every synchronized
        # optimizer update.  ``warmup_steps`` is exposed as global optimizer
        # steps, so scale the underlying scheduler's tick count accordingly.
        # With split batches enabled, the scheduler advances only once.
        scheduler_step_multiplier = 1
        if (
            bool(getattr(self.accelerator, "step_scheduler_with_optimizer", True))
            and not bool(getattr(self.accelerator, "split_batches", False))
        ):
            scheduler_step_multiplier = int(
                getattr(self.accelerator, "num_processes", 1)
            )
        scheduler_step_multiplier = max(scheduler_step_multiplier, 1)
        self.scheduler_warmup_steps = self.warmup_steps * scheduler_step_multiplier

        def lr_lambda(step: int) -> float:
            if self.scheduler_warmup_steps <= 0:
                return 1.0
            return min(1.0, float(step + 1) / float(self.scheduler_warmup_steps))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.scheduler
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._last_saved_step = -1

    def _condition_rng(self):
        model = self.accelerator.unwrap_model(self.model)
        return getattr(model, "cond_rng", None)

    def load_state(self, checkpoint_dir: str | os.PathLike[str]) -> None:
        """Restore Accelerate state and the trainer counters.

        A safetensors path is handled by the entrypoint as a weights-only warm
        start; this method is for checkpoint-N folders.
        """
        checkpoint_dir = Path(checkpoint_dir)
        self.accelerator.load_state(str(checkpoint_dir))
        state_file = checkpoint_dir / "trainer_state.json"
        state = None
        if state_file.is_file():
            with state_file.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        if state is None:
            match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
            state = {"global_step": int(match.group(1)) if match else 0}
        self.global_step = int(state.get("global_step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.batch_in_epoch = int(state.get("batch_in_epoch", 0))
        encoded_rng = state.get("condition_rng_state")
        condition_rng = self._condition_rng()
        if encoded_rng and condition_rng is not None:
            condition_rng.setstate(_nested_tuple(encoded_rng))
        self._last_saved_step = self.global_step

    def _save(self) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        if self.accelerator.is_main_process:
            unwrapped = self.accelerator.unwrap_model(self.model)
            save_trainable_checkpoint(
                unwrapped,
                self.output_dir / f"step-{self.global_step}.safetensors",
                metadata={"global_step": self.global_step},
            )
        self.accelerator.wait_for_everyone()
        # The engine state is useful for local resume but is not part of the
        # public model artifact.
        save_accelerate_state(self.accelerator, checkpoint_dir)
        if self.accelerator.is_main_process:
            condition_rng = self._condition_rng()
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
            }
            if condition_rng is not None:
                state["condition_rng_state"] = condition_rng.getstate()
            with (checkpoint_dir / "trainer_state.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
        self.accelerator.wait_for_everyone()
        self._last_saved_step = self.global_step

    def train(self, *, num_epochs: int = 1) -> int:
        num_epochs = max(int(num_epochs), 1)
        if self.global_step >= self.max_steps:
            return self.global_step
        self.model.train()
        start_epoch = self.epoch
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            if hasattr(self.dataloader, "set_epoch"):
                self.dataloader.set_epoch(epoch)
            elif hasattr(getattr(self.dataloader, "sampler", None), "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)

            skip_batches = self.batch_in_epoch if epoch == start_epoch else 0
            active_dataloader = self.dataloader
            if skip_batches:
                if hasattr(self.accelerator, "skip_first_batches"):
                    active_dataloader = self.accelerator.skip_first_batches(
                        self.dataloader, skip_batches
                    )
                else:
                    active_dataloader = islice(
                        self.dataloader, skip_batches, None
                    )

            for batch_index, batch in enumerate(
                active_dataloader, start=skip_batches
            ):
                with self.accelerator.accumulate(self.model):
                    result = self.model(batch)
                    if isinstance(result, tuple):
                        _, loss = result
                    else:
                        loss = result
                    if isinstance(loss, Mapping):
                        loss = loss["loss"]
                    finite = torch.isfinite(loss.detach()).all().to(
                        dtype=torch.int32
                    )
                    finite = self.accelerator.reduce(finite, reduction="min")
                    if not bool(finite.item()):
                        raise FloatingPointError(
                            f"non-finite training loss at step "
                            f"{self.global_step}: {loss}"
                        )
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        distributed_name = getattr(
                            self.accelerator.distributed_type, "name", ""
                        )
                        if (
                            self.max_grad_norm is not None
                            and distributed_name != "DEEPSPEED"
                        ):
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), float(self.max_grad_norm)
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                self.batch_in_epoch = batch_index + 1
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.global_step % self.log_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        self.accelerator.print(
                            f"step={self.global_step} loss={float(loss.detach()):.6f} lr={lr:.6g}"
                        )
                    if self.global_step % self.save_steps == 0:
                        self._save()
                    if self.global_step >= self.max_steps:
                        if self._last_saved_step != self.global_step:
                            self._save()
                        return self.global_step

            self.epoch = epoch + 1
            self.batch_in_epoch = 0

        self.accelerator.wait_for_everyone()
        if self.global_step > 0 and self._last_saved_step != self.global_step:
            self._save()
        return self.global_step


__all__ = ["KairosTrainer"]
