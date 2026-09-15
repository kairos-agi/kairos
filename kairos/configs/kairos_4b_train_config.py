"""Kairos 4B target-shape training configuration.

The dataset is sampled to 832x480 clips with 81 frames at 16 FPS. Source
videos may use other resolutions, frame counts, and frame rates. The default
condition distribution is 100% TI2V; adjust ``cond_probs`` to enable other modes.
"""

import os


KAIROS_MODEL_DIR = os.environ.get("KAIROS_MODEL_DIR", "models")
PRETRAINED_DIT = os.environ.get(
    "KAIROS_VIDEO_DIT",
    f"{KAIROS_MODEL_DIR}/Kairos3.1-4B-robot-480P/kairos-4B-robot-3.1-480P.safetensors",
)
VAE_PATH = os.environ.get(
    "KAIROS_VAE",
    f"{KAIROS_MODEL_DIR}/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
)
TEXT_ENCODER_PATH = os.environ.get(
    "KAIROS_TEXT_ENCODER",
    f"{KAIROS_MODEL_DIR}/Qwen/Qwen3.5-2B/",
)


cond_names = ["t2v", "ti2v", "i2v", "null2v"]
cond_probs = [0.0, 1.0, 0.0, 0.0]
seed = 42

pipeline = dict(
    type="KairosEmbodiedAPI",
    exec_mode="train",
    seed=seed,
    pipeline_type="KairosEmbodiedWAMPipeline",
    training_args=dict(
        cond_names=cond_names,
        cond_probs=cond_probs,
        trainable_models=["dit.video_dit"],
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        gradient_checkpointing_level="block",
        min_timestep_boundary=0.0,
        max_timestep_boundary=1.0,
    ),
    pipeline_args=dict(
        load_dit_fn="strict_load",
        vae_path=VAE_PATH,
        text_encoder_config=dict(
            type="Qwen3_5_TextEncoder",
            from_pretrained=TEXT_ENCODER_PATH,
        ),
        dit_config=dict(
            dit_type="Simple_Multi_DIT_Wrapper",
            video_dit=dict(
                dit_type="KairosDiTV2",
                pretrained_path=PRETRAINED_DIT,
                has_image_input=False,
                patch_size=[1, 2, 2],
                in_dim=16,
                dim=2560,
                ffn_dim=10240,
                freq_dim=256,
                text_dim=2048,
                out_dim=16,
                num_heads=20,
                num_layers=32,
                layers_settings=[
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                    "SWA", "SWA", "DSWA", "GATED",
                ],
                eps=1e-6,
                seperated_timestep=True,
                require_clip_embedding=False,
                require_vae_embedding=False,
                fuse_vae_embedding_in_latents=True,
                attn_method="flex",
                restrict_history_query_to_history=False,
                dilated_lengths=[4],
                use_first_frame_cond=False,
                use_seq_parallel=False,
                use_tp_in_getaeddeltanet=False,
                use_tp_in_self_attn=False,
                attend_k0=False,
            ),
        ),
    ),
)


data = dict(
    train=dict(
        manifest=os.environ.get("KAIROS_VIDEO_MANIFEST", "data/train.jsonl"),
        data_root=os.environ.get("KAIROS_VIDEO_ROOT", "data"),
        width=832,
        height=480,
        num_frames=81,
        fps=16.0,  # target sampling FPS
    )
)

data_loader = dict(
    train=dict(
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
)

trainer_cfg = dict(
    output_dir=os.environ.get("KAIROS_OUTPUT_DIR", "runs/kairos_train"),
    learning_rate=1e-4,
    weight_decay=0.0,
    betas=(0.9, 0.95),
    warmup_steps=500,
    max_steps=1000,
    num_epochs=2000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    save_steps=1000,
    log_steps=10,
    resume="",
)
