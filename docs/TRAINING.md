# Kairos training

This guide covers the Kairos Flow Matching training entrypoint, its target-shape
local video dataset, and the Accelerate launch options.

## Overview

The training path provides:

- a UTF-8 JSONL manifest with local video clips and captions;
- one output shape per run (832x480, 81 frames, 16 FPS by default);
- configurable `t2v`, `ti2v`, `i2v`, and `null2v` conditions;
- Accelerate DDP or DeepSpeed ZeRO-1 execution;
- trainable-weight export and run-state checkpoints.

The API constructs `KairosFlowMatchingModel`. `KairosTrainer` updates the
modules listed by `trainable_models` (the default is `dit.video_dit`), while
the VAE, text encoder, image encoder, motion/VACE components, and action head
remain in evaluation mode. The trainer does not apply EMA or an action loss.

The inference environment supplies the model kernels. Install the base
requirements first, then add the packages in `requirements-train.txt`. A
CUDA-capable PyTorch installation is required for a practical run.

## Install

    pip install -r requirements.txt
    pip install -r requirements-train.txt

Set `KAIROS_MODEL_DIR` or individual model variables when checkpoints are not
under `models`:

    KAIROS_MODEL_DIR=models
    KAIROS_VIDEO_DIT=models/Kairos3.1-4B-robot-480P/kairos-4B-robot-3.1-480P.safetensors
    KAIROS_VAE=models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
    KAIROS_TEXT_ENCODER=models/Qwen/Qwen3.5-2B

## Manifest and clip contract

`SimpleVideoDataset` reads one JSON object per non-empty line. The object must
contain a relative `video` path and a string in `caption` or `prompt`:

    {"video": "clips/example.mp4", "caption": "a robot moves the cup"}

URI schemes, absolute paths, traversal outside the data root, and symlinks that
resolve outside the data root are rejected. Each source clip may have any
resolution, frame count, and frame rate, provided it is long enough to cover the
target sampling interval. The dataset lazily samples the target number of frames
at the configured target FPS: high-FPS sources are downsampled and low-FPS
sources reuse source frames as needed. Every sampled frame is directly resized
to the target width and height (without aspect-ratio preservation or center
cropping). The default output is 832x480, 81 frames at 16 FPS.

Frames are passed to the pipeline as nested PIL frame lists. There is no
temporal bucket selection. A typical directory is:

    data/
      train.jsonl
      clips/
        example.mp4

The example schema is in `examples/train_metadata.example.jsonl`. Add records
when a larger batch or multiple processes are used.

## Conditions

The API samples one condition for each training batch:

- `t2v`: caption conditioning without a first-image latent;
- `ti2v`: caption conditioning plus the first frame;
- `i2v`: the first frame with an empty caption;
- `null2v`: no image and a zeroed text context.

Set the values in `pipeline.training_args` in
`kairos/configs/kairos_4b_train_config.py`:

    cond_names = ["t2v", "ti2v", "i2v", "null2v"]
    cond_probs = [0.0, 1.0, 0.0, 0.0]

Probabilities must be non-negative and sum to one. `null2v` is the accepted
spelling inherited from the A-repository implementation; `n2v` is not accepted.

## Launch

The configuration reads these environment variables:

    KAIROS_VIDEO_MANIFEST=data/train.jsonl
    KAIROS_VIDEO_ROOT=data
    KAIROS_OUTPUT_DIR=runs/kairos_train

Run one process with the included launcher:

    bash examples/train_kairos.sh

The launcher uses bf16 and ZeRO-1 with one process by default. Set
`NUM_PROCESSES` on a multi-GPU host:

    NUM_PROCESSES=8 bash examples/train_kairos.sh

The values for `gradient_accumulation_steps` and
`max_grad_norm`/`gradient_clipping` must agree between the training config and
the DeepSpeed YAML. ZeRO-2/3 and FSDP are not supported by this entrypoint.
`warmup_steps` is counted in synchronized global optimizer updates; the trainer
automatically compensates for Accelerate's per-process scheduler ticks on
multi-GPU runs.

The VAE preprocessing defaults to `tiled=False`, matching the training loss
path. Reduce the batch size or adjust the local pipeline if the VAE does not fit
in memory.

## Checkpoints and resume

At a save step the trainer writes:

- `step-N.safetensors`: trainable `video_dit` weights for a weights warm-start;
- `checkpoint-N/`: Accelerate optimizer, scheduler, RNG, and trainer run state.

Warm-start from trainable weights:

    python examples/train_kairos.py --config kairos/configs/kairos_4b_train_config.py \
      --resume runs/kairos_train/step-1000.safetensors

Resume a complete run:

    python examples/train_kairos.py --config kairos/configs/kairos_4b_train_config.py \
      --resume runs/kairos_train/checkpoint-1000

The run state records the optimizer step, epoch/batch offset, condition RNG,
and the seedable data-sampler configuration so a resumed epoch can skip
consumed batches.
