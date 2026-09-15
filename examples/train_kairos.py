"""Launch Kairos training on a target-shape video dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine import Config
from torch.utils.data import DataLoader

from kairos.apis.builder import build_model_pipeline
from kairos.data import SimpleVideoDataset, collate_video_batch
from kairos.training import KairosTrainer, load_trainable_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kairos")
    parser.add_argument(
        "--config",
        default="kairos/configs/kairos_4b_train_config.py",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed

    cfg = Config.fromfile(args.config)
    train_cfg = cfg.trainer_cfg
    data_cfg = cfg.data.train
    loader_cfg = cfg.data_loader.train

    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    seed = int(cfg.get("seed", 42))
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=seed,
        ),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    set_seed(seed, device_specific=True)

    manifest = args.manifest or data_cfg.manifest
    data_root = args.data_root or data_cfg.data_root
    output_dir = Path(args.output_dir or train_cfg.output_dir)

    dataset = SimpleVideoDataset(
        manifest=manifest,
        data_root=data_root,
        width=int(data_cfg.get("width", 832)),
        height=int(data_cfg.get("height", 480)),
        num_frames=int(data_cfg.get("num_frames", 81)),
        fps=float(data_cfg.get("fps", 16.0)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=bool(loader_cfg.get("shuffle", True)),
        num_workers=int(loader_cfg.get("num_workers", 4)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", True)),
        collate_fn=collate_video_batch,
    )

    pipeline = build_model_pipeline(cfg.pipeline)
    if not hasattr(pipeline, "model"):
        raise RuntimeError("training config did not build pipeline.model; set exec_mode='train'")
    model = pipeline.model

    resume = args.resume if args.resume is not None else train_cfg.get("resume", "")
    if resume and str(resume).endswith(".safetensors"):
        load_trainable_checkpoint(model, resume, strict=True)

    trainer = KairosTrainer(
        model,
        dataloader,
        output_dir=output_dir,
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        max_steps=int(train_cfg.get("max_steps", 1000)),
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        save_steps=int(train_cfg.get("save_steps", 1000)),
        log_steps=int(train_cfg.get("log_steps", 10)),
        accelerator=accelerator,
    )

    if resume and not str(resume).endswith(".safetensors"):
        trainer.load_state(resume)

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"dataset size: {len(dataset)}")
        print(f"output dir: {output_dir}")
    trainer.train(num_epochs=int(train_cfg.get("num_epochs", 1)))


if __name__ == "__main__":
    main()
