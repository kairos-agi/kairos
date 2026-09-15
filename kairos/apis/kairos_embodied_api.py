"""Kairos model API and Flow Matching training wrapper."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Mapping

import torch
from mmengine import Config as MMConfig

from kairos.apis.builder import DITS, KAIROS_PROCESSOR, PIPELINES_API
from kairos.modules.utils import load_state_dict


_VIDEO_CONDITIONS = ("t2v", "ti2v", "i2v", "null2v")


def _validate_conditions(names, probs):
    names = list(_VIDEO_CONDITIONS if names is None else names)
    if not names:
        raise ValueError("condition names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"condition names must be unique: {names}")
    if probs is None:
        probs = [1.0 if name == "ti2v" else 0.0 for name in names]
        if not any(probs):
            probs[0] = 1.0
    probs = [float(value) for value in probs]
    if len(names) != len(probs):
        raise ValueError("condition names and probabilities must have the same length")
    if any(name not in _VIDEO_CONDITIONS for name in names):
        raise ValueError(f"supported conditions are {_VIDEO_CONDITIONS}, got {names}")
    if any(not math.isfinite(value) or value < 0 for value in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"condition probabilities must be non-negative and sum to one: {probs}")
    return names, probs


class KairosFlowMatchingModel(torch.nn.Module):
    """Flow Matching training wrapper around a Kairos pipeline."""

    def __init__(
        self,
        pipe: torch.nn.Module,
        *,
        trainable_models=("dit.video_dit",),
        cond_names=("t2v", "ti2v", "i2v", "null2v"),
        cond_probs=(0.0, 1.0, 0.0, 0.0),
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        gradient_checkpointing_level="block",
        min_timestep_boundary=0.0,
        max_timestep_boundary=1.0,
        seed=42,
    ):
        super().__init__()
        self.pipe = pipe
        self.trainable_model_names = self._normalize_trainable_models(trainable_models)
        self.cond_names, self.cond_probs = _validate_conditions(cond_names, cond_probs)
        self.cond_rng = random.Random(int(seed))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.gradient_checkpointing_level = gradient_checkpointing_level
        self.min_timestep_boundary = float(min_timestep_boundary)
        self.max_timestep_boundary = float(max_timestep_boundary)

        if not 0.0 <= self.min_timestep_boundary < self.max_timestep_boundary <= 1.0:
            raise ValueError("invalid timestep boundaries")
        if self.gradient_checkpointing_level not in ("block", "op"):
            raise ValueError("gradient_checkpointing_level must be 'block' or 'op'")

        self.pipe.requires_grad_(False)
        for name in self.trainable_model_names:
            self._get_trainable_model(name).requires_grad_(True)
        if not any(parameter.requires_grad for parameter in self.pipe.parameters()):
            raise ValueError("trainable_models did not select any parameters")
        self._set_component_modes(mode=True)

    @staticmethod
    def _normalize_trainable_models(trainable_models):
        if isinstance(trainable_models, str):
            trainable_models = trainable_models.split(",")
        if not isinstance(trainable_models, (list, tuple)):
            raise TypeError("trainable_models must be a string, list, or tuple")
        names = [str(name).strip() for name in trainable_models if str(name).strip()]
        if not names:
            raise ValueError("trainable_models must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"trainable_models must be unique: {names}")
        return tuple(names)

    def _get_trainable_model(self, name: str) -> torch.nn.Module:
        try:
            module = self.pipe.get_submodule(name)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"unknown trainable model path: {name}") from exc
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"trainable model path is not a module: {name}")
        return module

    def _set_component_modes(self, mode: bool) -> None:
        self.pipe.eval()
        for name in self.trainable_model_names:
            self._get_trainable_model(name).train(mode)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_component_modes(mode=mode)
        return self

    def trainable_modules(self):
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def trainable_param_names(self):
        return {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def export_trainable_state_dict(self, state_dict):
        trainable_param_names = self.trainable_param_names()
        return {
            name: value
            for name, value in state_dict.items()
            if name in trainable_param_names
        }

    def _choose_condition(self, data: Mapping[str, Any]) -> str:
        requested = data.get("condition_type")
        if isinstance(requested, (list, tuple)):
            values = list(dict.fromkeys(requested))
            if len(values) > 1:
                raise ValueError("a training batch must use one condition_type")
            requested = values[0] if values else None
        if requested is not None:
            if requested not in _VIDEO_CONDITIONS:
                raise ValueError(f"unsupported condition_type: {requested}")
            return requested
        return self.cond_rng.choices(self.cond_names, weights=self.cond_probs, k=1)[0]

    def forward_preprocess(self, data: Mapping[str, Any]):
        videos = data.get("video")
        prompts = data.get("prompt")
        if not isinstance(videos, (list, tuple)) or not videos:
            raise ValueError("training batch must contain a non-empty video list")
        if not isinstance(prompts, (list, tuple)) or len(prompts) != len(videos):
            raise ValueError("training batch prompt/video batch sizes differ")
        if any(
            not isinstance(video, (list, tuple)) or not video
            for video in videos
        ):
            raise ValueError("training batch contains an empty or malformed video")
        frame_counts = {len(video) for video in videos}
        frame_sizes = {
            getattr(frame, "size", None)
            for video in videos
            for frame in video
        }
        if len(frame_counts) != 1 or len(frame_sizes) != 1 or None in frame_sizes:
            raise ValueError(
                "all videos in a batch must have the same frame shape and length"
            )

        condition = self._choose_condition(data)
        first_images = [video[0] for video in videos]
        num_frames = min(len(video) for video in videos)
        if num_frames <= 0:
            raise ValueError("training batch contains an empty video")
        height, width = first_images[0].height, first_images[0].width

        self.pipe.scheduler.training = True
        inputs_shared = {
            "cfg_scale": 1,
            "cfg_merge": False,
            "input_video": videos,
            "input_image": first_images if condition in ("ti2v", "i2v") else None,
            "end_image": None,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "batch_size": len(videos),
            "seed": None,
            "rand_device": str(self.pipe.device),
            "tiled": False,
            "tile_size": (30, 52),
            "tile_stride": (15, 26),
            "vace_reference_image": None,
            "robot_action_horizon": None,
            "input_audio": None,
            "audio_embeds": None,
            "s2v_pose_video": None,
            "s2v_pose_latents": None,
            "motion_video": None,
            "control_video": None,
            "reference_image": None,
            "camera_control_direction": None,
            "camera_control_speed": None,
            "camera_control_origin": None,
            "vace_video": None,
            "vace_video_mask": None,
            "vace_scale": 1.0,
            "motion_bucket_id": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "gradient_checkpointing_level": self.gradient_checkpointing_level,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        prompt_values = list(prompts)
        if condition == "i2v":
            prompt_values = ["" for _ in prompt_values]
        inputs_posi = {"prompt": prompt_values, "positive": True}
        inputs_nega = {}

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        if condition == "null2v":
            context = inputs_posi.get("context")
            if context is not None:
                context.zero_()
            context_mask = inputs_posi.get("context_mask")
            if context_mask is not None:
                context_mask.zero_()

        merged = dict(inputs_shared)
        merged.update(inputs_posi)
        merged["condition_type"] = condition
        return merged

    def forward(self, data, inputs=None):
        if inputs is None:
            with torch.no_grad():
                inputs = self.forward_preprocess(data)
        loss = self.pipe.training_loss(dit=self.pipe.dit, **inputs)
        output = {
            "loss": loss,
            "condition_type": inputs.get("condition_type", ""),
        }
        return output, loss

    def training_step(self, data_batch, data_batch_idx=None):
        return self.forward(data_batch)


@PIPELINES_API.register_module()
class KairosEmbodiedAPI(torch.nn.Module):
    """Model API supporting inference and training execution modes."""

    def __init__(
        self,
        config=MMConfig(
            dict(
                exec_mode="infer",
                pipeline_type="KairosEmbodiedPipeline",
                pretrained_dit=None,
                vae_path=None,
                text_encoder_path=None,
                pipeline_args=None,
                training_args=None,
                tea_cache_l1_thresh=None,
                tea_cache_model_id="",
            )
        ),
        torch_dtype=torch.bfloat16,
        device="cuda",
    ):
        if isinstance(torch_dtype, str):
            aliases = {
                "bf16": "bfloat16",
                "fp16": "float16",
                "fp32": "float32",
            }
            torch_dtype = getattr(
                torch, aliases.get(torch_dtype.lower(), torch_dtype)
            )
        super().__init__()
        self._init_config = config
        self.exec_mode = str(config.get("exec_mode", "infer")).lower()
        if self.exec_mode == "test":
            self.exec_mode = "infer"
        if self.exec_mode not in ("infer", "train"):
            raise ValueError("exec_mode must be 'infer' or 'train'")

        self.tea_cache_l1_thresh = config.get("tea_cache_l1_thresh", None)
        self.tea_cache_model_id = config.get("tea_cache_model_id", "")
        self.parallel_mode = config.get("parallel_mode", None)
        pretrained_dit = config.get("pretrained_dit", None)
        pipeline_type = config.get("pipeline_type", "KairosEmbodiedPipeline")
        if (
            self.exec_mode == "train"
            and pipeline_type != "KairosEmbodiedWAMPipeline"
        ):
            raise ValueError(
                "exec_mode='train' is supported only with "
                "KairosEmbodiedWAMPipeline"
            )
        pipeline_args = copy.deepcopy(config.get("pipeline_args", {}) or {})
        training_args = copy.deepcopy(config.get("training_args", {}) or {})
        if not isinstance(training_args, Mapping):
            raise TypeError("training_args must be a mapping")
        # Populate registries for callers that instantiate this API directly.
        import kairos.modules.dits  # noqa: F401
        import kairos.pipelines  # noqa: F401

        dit_config = copy.deepcopy(pipeline_args.pop("dit_config", None))
        load_dit_fn = pipeline_args.pop("load_dit_fn", None)
        pipeline_args["parallel_mode"] = self.parallel_mode
        # Training policy belongs to this wrapper, not the underlying pipeline.
        training_arg_names = (
            "trainable_models",
            "cond_names",
            "cond_probs",
            "use_gradient_checkpointing",
            "use_gradient_checkpointing_offload",
            "gradient_checkpointing_level",
            "min_timestep_boundary",
            "max_timestep_boundary",
        )
        for name in training_arg_names:
            if name in pipeline_args:
                training_args.setdefault(name, pipeline_args.pop(name))
        pipeline_args.pop("exec_mode", None)

        if dit_config:
            print("Init KairosDiT model with config: ", dit_config)
            dit_type = dit_config.pop("dit_type")
            dit_cls = DITS.get(dit_type)
            dit = dit_cls(**dit_config)
            total_params = sum(p.numel() for p in dit.parameters()) / 1e9
            print(f"Total parameters of DiT: {total_params:.3f} B")
            if pretrained_dit:
                if load_dit_fn != "strict_load":
                    raise NotImplementedError(
                        f"unsupported load_dit_fn: {load_dit_fn}"
                    )
                state_dict = load_state_dict(pretrained_dit)
                dit.load_state_dict(state_dict, strict=True)
            dit = dit.to(device=device, dtype=torch_dtype)
            pipeline_args["dit"] = dit

        pipeline_cls = KAIROS_PROCESSOR.get(pipeline_type)
        pipe = pipeline_cls.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            **pipeline_args,
        )

        if self.exec_mode == "train":
            self.model = KairosFlowMatchingModel(
                pipe,
                trainable_models=training_args.get(
                    "trainable_models", ("dit.video_dit",)
                ),
                cond_names=training_args.get(
                    "cond_names", ("t2v", "ti2v", "i2v", "null2v")
                ),
                cond_probs=training_args.get(
                    "cond_probs", (0.0, 1.0, 0.0, 0.0)
                ),
                use_gradient_checkpointing=training_args.get(
                    "use_gradient_checkpointing", True
                ),
                use_gradient_checkpointing_offload=training_args.get(
                    "use_gradient_checkpointing_offload", False
                ),
                gradient_checkpointing_level=training_args.get(
                    "gradient_checkpointing_level", "block"
                ),
                min_timestep_boundary=training_args.get(
                    "min_timestep_boundary", 0.0
                ),
                max_timestep_boundary=training_args.get(
                    "max_timestep_boundary", 1.0
                ),
                seed=config.get("seed", 42),
            )
        else:
            self._pipe = pipe

        total_params = sum(p.numel() for p in pipe.parameters()) / 1e9
        print(f"Total parameters of the whole model: {total_params:.3f} B")

    @property
    def pipe(self):
        return self.model.pipe if self.exec_mode == "train" else self._pipe

    def trainable_modules(self):
        if self.exec_mode != "train":
            raise RuntimeError(
                "trainable_modules is only available in exec_mode='train'"
            )
        return self.model.trainable_modules()

    def forward(self, *args, **kwargs):
        if self.exec_mode == "train":
            return self.model(*args, **kwargs)
        kwargs["tea_cache_l1_thresh"] = self.tea_cache_l1_thresh
        kwargs["tea_cache_model_id"] = self.tea_cache_model_id
        kwargs["parallel_mode"] = self.parallel_mode
        return self.pipe(**kwargs)


__all__ = ["KairosEmbodiedAPI", "KairosFlowMatchingModel"]
