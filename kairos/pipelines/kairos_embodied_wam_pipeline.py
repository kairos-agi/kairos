import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from .base_pipeline import BasePipeline, PipelineUnit, PipelineUnitRunner
from kairos.modules.utils import load_state_dict, init_weights_on_device
from kairos.modules.dits import sinusoidal_embedding_1d
from kairos.modules.vaes import WanVideoVAE
from kairos.modules.text_encoders import QwenVLTextEncoder
from kairos.modules.schedulers.flow_match import FlowMatchScheduler

from kairos.apis.builder import KAIROS_PROCESSOR

import torch.distributed as dist
from kairos.modules.utils import parallel_state
from kairos.modules.vaes.parallel_vae_wrapper import ParallelVAEWrapper
from kairos.modules.utils.tp_utils import _gather_input_sp, _distribute_input_sp


from kairos.modules.dits.mot import (
    mot_pre_video_dit, mot_pre_action_dit,
    mot_post_video_dit, mot_post_action_dit,
    mot_dit_blocks_forward, mot_infer_pure_video_dit_once,
    mot_infer_pure_action_dit_once_with_video_cache,
    mot_infer_joint_video_action_dit_once,
)
from kairos.modules.dits.mot_mask_utils import build_mot_attention_mask_info_dict_wrapper


@KAIROS_PROCESSOR.register_module()
class KairosEmbodiedWAMPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, vram_management_enabled = False, text_encoder_config=dict()):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.text_encoder_config = text_encoder_config
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True,
                                            exponential_shift=True, exponential_shift_mu=1.609)
        self.action_shift = 5.0
        self.scheduler_action = FlowMatchScheduler(shift=self.action_shift, sigma_min=0.0, extra_one_step=True,
                                            exponential_shift=False, exponential_shift_mu=None)
        self.prompter =  None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),

            # kairos robot state encoder
            WanVideoUnit_RobotStateEncodeToContext(),

            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video

        self.use_dpo = False
        self.ref_dit = None

        self.with_action_head = False
        # self.wam_ar_cfg = normalize_wam_ar_cfg()
        self.text_encoder_config = text_encoder_config

        # *******************************************************
        # timestep seed
        # self.timestep_seed = get_timestep_seed()

        # self.timestep_rng = None
        # end timestep seed
        # *******************************************************

    def load_lora(self, module, path, alpha=1):
        raise NotImplementedError()

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        dit = None,
        vae_path = None,
        text_encoder_config=dict(),
        vram_management_enabled = False,
        parallel_mode = None,
    ):
        # ***********************************************************
        # Initialize pipeline
        pipe = KairosEmbodiedWAMPipeline(device=device, torch_dtype=torch_dtype, text_encoder_config = text_encoder_config)

        # ***********************************************************
        # load text_encoder
        assert len(text_encoder_config) > 0, "text_encoder_config is empty"
        # Load models
        text_encoder_type = text_encoder_config.pop('type')
        if text_encoder_type == 'Qwen3_5_TextEncoder':
            from kairos.modules.text_encoders import Qwen3_5_TextEncoder
            text_encoder = Qwen3_5_TextEncoder(dtype=torch_dtype, device=device, **text_encoder_config)
        elif text_encoder_type == 'QwenVLTextEncoder':
            from kairos.modules.text_encoders import QwenVLTextEncoder
            text_encoder = QwenVLTextEncoder(dtype=torch_dtype, device=device, **text_encoder_config)
        else:
            raise NotImplementedError("Not implemented for non-qwen text encoder")
        pipe.prompter = text_encoder

        # ***********************************************************
        # load dit
        assert dit is not None
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit

        # ***********************************************************
        # load WanVideoVAE 

        vae_model_cls = WanVideoVAE
        print(f'loading vae from path:  {vae_path} ...')
        vae_state_dict = load_state_dict(vae_path)
        state_dict_converter = vae_model_cls.state_dict_converter()
        state_dict_results = state_dict_converter.from_civitai(vae_state_dict)
        extra_kwargs = {}
        with init_weights_on_device():
            vae_ = vae_model_cls(**extra_kwargs)
        if hasattr(vae_, "eval"):
            vae_ = vae_.eval()
        vae_.load_state_dict(state_dict_results, assign=True)
        vae_ = vae_.to(dtype=torch_dtype, device=device)
        pipe.vae = vae_
        vae_group = parallel_state.get_vae_parallel_group()
        vae_size = dist.get_world_size(vae_group) if (vae_group is not None and dist.is_initialized()) else 1
        if vae_size > 1 and parallel_mode:
            if (parallel_mode == "cp" or parallel_mode == "sp") and not hasattr(pipe.vae, 'parallel_group'): 
                print(f">>>> Wrapping VAE for Context Parallel Decoding with cp_size:{vae_size}, parallel mode: {parallel_mode} <<<<")
                pipe.vae = ParallelVAEWrapper(pipe.vae, vae_group, parallel_mode)
        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        print('loading vae done')

        print(f'loading KairosEmbodiedWAMPipeline done .')
        return pipe

    def _solver_step(self, base_scheduler, solver_scheduler, model_output, progress_id, sample):
        if solver_scheduler is None:
            return base_scheduler.step(model_output, base_scheduler.timesteps[progress_id], sample)

        sample_fp32 = sample.to(dtype=torch.float32)
        model_output_fp32 = model_output.to(dtype=torch.float32)
        prev_sample = solver_scheduler.step(
            model_output_fp32,
            solver_scheduler.timesteps[progress_id],
            sample_fp32,
        ).prev_sample
        return prev_sample.to(dtype=sample.dtype)

    @staticmethod
    def _get_conditioning(inputs_shared, inputs_posi, cfg_merge, batch_size=None):
        if not cfg_merge:
            return inputs_posi["context"], inputs_posi["context_mask"]

        context = inputs_shared["context"]
        context_mask = inputs_shared.get("context_mask")
        if batch_size is not None:
            context = context[:batch_size]
            if context_mask is not None:
                context_mask = context_mask[:batch_size]
        return context, context_mask

    @torch.no_grad()
    def infer_video_action_joint(
        self, 
        inputs_shared,
        inputs_posi,
        inputs_nega,
        switch_DiT_boundary,
        progress_bar_cmd,
        cfg_scale,
        cfg_merge,
        tiled,
        tile_size,
        tile_stride,
        solver_video=None,
        solver_action=None,
    ):

        assert len(inputs_shared["latents"].shape) == 5
        # [B, C, T, H, W]
        b,c,t,h,w = inputs_shared["latents"].shape
        video_seq_len = t * h * w // self.dit.video_dit.patch_size[1] // self.dit.video_dit.patch_size[2]
        video_tokens_per_frame = h * w // self.dit.video_dit.patch_size[1] // self.dit.video_dit.patch_size[2]
        if "first_frame_latents" in inputs_shared:
            num_history_frames = inputs_shared["first_frame_latents"].shape[2]
        else:
            num_history_frames = 0
        
        action_seq_len = inputs_shared["noise_action"].shape[1]

        attention_mask_d = build_mot_attention_mask_info_dict_wrapper(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_history_frames=num_history_frames,
            device=inputs_shared["latents"].device,
            mask_types=['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
            all_atten_history = self.dit.video_dit.all_atten_history,
            atten_modes = self.dit.video_dit.atten_modes,
            window_size=self.dit.video_dit.window_size,
            dilated_length=self.dit.video_dit.dilated_length,
            flex_block_size=self.dit.video_dit.flex_block_size,
            attn_method=self.dit.video_dit.attn_method,
            restrict_history_query_to_history=self.dit.video_dit.restrict_history_query_to_history,
        )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        context, context_mask = self._get_conditioning(
            inputs_shared,
            inputs_posi,
            cfg_merge,
            batch_size=(
                inputs_shared["latents"].shape[0] if cfg_scale == 1.0 else None
            ),
        )

        assert len(self.scheduler.timesteps) == len(self.scheduler_action.timesteps)

        for progress_id, (timestep, timestep_action) in enumerate(progress_bar_cmd(zip(self.scheduler.timesteps, self.scheduler_action.timesteps), total=len(self.scheduler.timesteps))):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            timestep_action = timestep_action.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference

            noise_pred_posi, noise_pred_action_posi = mot_infer_joint_video_action_dit_once(
                video_dit=self.dit.video_dit,
                action_dit=self.dit.action_dit,
                
                context=context,
                context_mask=context_mask,
                attention_mask_d=attention_mask_d,

                latents=inputs_shared["latents"],
                timestep=timestep,
                fuse_vae_embedding_in_latents=inputs_shared.get("fuse_vae_embedding_in_latents", False),
                first_frame_latents=inputs_shared.get("first_frame_latents", None),
                action_latents=inputs_shared["noise_action"],
                timestep_action=timestep_action,
            )
        

            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    # The unmerged path predicts actions from positive conditioning only.
                    noise_pred_action_posi = noise_pred_action_posi.chunk(2, dim=0)[0]
                else:
                    noise_pred_nega =mot_infer_pure_video_dit_once(
                        video_dit=self.dit.video_dit,
                        latents=inputs_shared["latents"],
                        first_frame_latents=inputs_shared.get("first_frame_latents", None),
                        fuse_vae_embedding_in_latents=inputs_shared.get("fuse_vae_embedding_in_latents", False),
                        timestep = timestep,
                        context=inputs_nega["context"],
                        context_mask=inputs_nega["context_mask"],
                        attention_mask_d=attention_mask_d,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            noise_pred_action = noise_pred_action_posi

            # Scheduler
            inputs_shared["latents"] = self._solver_step(
                self.scheduler,
                solver_video,
                noise_pred,
                progress_id,
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                his_t = inputs_shared["first_frame_latents"].shape[2]
                inputs_shared["latents"][:, :, 0:his_t] = inputs_shared["first_frame_latents"]
            


            inputs_shared["noise_action"] = self._solver_step(
                self.scheduler_action,
                solver_action,
                noise_pred_action,
                progress_id,
                inputs_shared["noise_action"],
            )
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)

        action = inputs_shared["noise_action"].detach()

        return {
            "video": video,
            "action": action,
        }


    @torch.no_grad()
    def infer_video(
        self, 
        inputs_shared,
        inputs_posi,
        inputs_nega,
        switch_DiT_boundary,
        progress_bar_cmd,
        cfg_scale,
        cfg_merge,
        tiled,
        tile_size,
        tile_stride,
        solver_video=None,
    ):

        assert len(inputs_shared["latents"].shape) == 5
        # [B, C, T, H, W]
        b,c,t,h,w = inputs_shared["latents"].shape
        video_seq_len = t * h * w // self.dit.video_dit.patch_size[1] // self.dit.video_dit.patch_size[2]
        video_tokens_per_frame = h * w // self.dit.video_dit.patch_size[1] // self.dit.video_dit.patch_size[2]
        if "first_frame_latents" in inputs_shared:
            num_history_frames = inputs_shared["first_frame_latents"].shape[2]
        else:
            num_history_frames = 0

        attention_mask_d = build_mot_attention_mask_info_dict_wrapper(
            video_seq_len=video_seq_len,
            action_seq_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            num_history_frames=num_history_frames,
            device=inputs_shared["latents"].device,
            mask_types=['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
            all_atten_history = self.dit.video_dit.all_atten_history,
            atten_modes = self.dit.video_dit.atten_modes,
            window_size=self.dit.video_dit.window_size,
            dilated_length=self.dit.video_dit.dilated_length,
            flex_block_size=self.dit.video_dit.flex_block_size,
            attn_method=self.dit.video_dit.attn_method,
            restrict_history_query_to_history=self.dit.video_dit.restrict_history_query_to_history,
        )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        context, context_mask = self._get_conditioning(
            inputs_shared,
            inputs_posi,
            cfg_merge,
            batch_size=(
                inputs_shared["latents"].shape[0] if cfg_scale == 1.0 else None
            ),
        )
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = mot_infer_pure_video_dit_once(
                video_dit=self.dit.video_dit,
                latents=inputs_shared["latents"],
                first_frame_latents=inputs_shared.get("first_frame_latents", None),
                fuse_vae_embedding_in_latents=inputs_shared.get("fuse_vae_embedding_in_latents", False),
                timestep = timestep,
                context=context,
                context_mask=context_mask,
                attention_mask_d=attention_mask_d,
            )
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega =mot_infer_pure_video_dit_once(
                        video_dit=self.dit.video_dit,
                        latents=inputs_shared["latents"],
                        first_frame_latents=inputs_shared.get("first_frame_latents", None),
                        fuse_vae_embedding_in_latents=inputs_shared.get("fuse_vae_embedding_in_latents", False),
                        timestep = timestep,
                        context=inputs_nega["context"],
                        context_mask=inputs_nega["context_mask"],
                        attention_mask_d=attention_mask_d,
                    )

                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self._solver_step(
                self.scheduler,
                solver_video,
                noise_pred,
                progress_id,
                inputs_shared["latents"],
            )

            if "first_frame_latents" in inputs_shared:
                his_t = inputs_shared["first_frame_latents"].shape[2]
                inputs_shared["latents"][:, :, 0:his_t] = inputs_shared["first_frame_latents"]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return {
            "video": video,
        }

    @torch.no_grad()
    def infer_action(
        self, 
        inputs_shared,
        inputs_posi,
        inputs_nega,
        switch_DiT_boundary,
        progress_bar_cmd,
        solver_action=None,
        cfg_merge=False,
    ):
        latents_action = inputs_shared['noise_action']
        context, context_mask = self._get_conditioning(
            inputs_shared,
            inputs_posi,
            cfg_merge,
            batch_size=latents_action.shape[0],
        )

        first_frame_latents = inputs_shared["first_frame_latents"]

        # *****************************************************
        # infer video-dit once || start

        # pre video dit
        video_pre = mot_pre_video_dit(dit=self.dit.video_dit,
                        first_frame_latents=first_frame_latents,
                        latents=inputs_shared["latents"], 
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=inputs_shared["fuse_vae_embedding_in_latents"]
                        )

        video_seq_len = int(video_pre["x"].shape[1])

        assert video_seq_len == first_frame_latents.shape[2] * video_pre["tokens_per_frame"]

        attention_mask_d = build_mot_attention_mask_info_dict_wrapper(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["tokens_per_frame"]),
            num_history_frames=first_frame_latents.shape[2],
            device=video_pre["x"].device,
            mask_types=['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
            all_atten_history = self.dit.video_dit.all_atten_history,
            atten_modes = self.dit.video_dit.atten_modes,
            window_size=self.dit.video_dit.window_size,
            dilated_length=self.dit.video_dit.dilated_length,
            flex_block_size=self.dit.video_dit.flex_block_size,
            attn_method=self.dit.video_dit.attn_method,
            restrict_history_query_to_history=self.dit.video_dit.restrict_history_query_to_history,
        )

        mot_kv_cache_out = mot_dit_blocks_forward(
            dit_d={
                "video": self.dit.video_dit,
            },
            attention_mask_d=attention_mask_d,
            embed_d={
                "video": video_pre['x'],
            },
            freqs_d={
                "video": video_pre['freqs'],
            },
            context_d={
                "video": {
                    "context": video_pre['context'],
                    "context_mask": video_pre['context_mask'],
                },
            },
            t_mod_d={
                "video": video_pre['t_mod'],
            },
            video_seq_len=video_seq_len,
            return_kv_cache=True,
        )
        video_kv_cache = mot_kv_cache_out['kv_cache']['video']

        # infer video-dit once || end
        # *****************************************************

        # action Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        for progress_id, timestep_action in enumerate(progress_bar_cmd(self.scheduler_action.timesteps)):
            # Switch DiT if necessary
            if timestep_action.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
            # Timestep
            timestep_action = timestep_action.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            
            pred_action_posi = mot_infer_pure_action_dit_once_with_video_cache(
                action_dit=self.dit.action_dit,
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask_d=attention_mask_d,
                video_kv_cache=video_kv_cache,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi
            latents_action = self._solver_step(
                self.scheduler_action,
                solver_action,
                pred_action,
                progress_id,
                latents_action,
            )

        return {
            "action": latents_action.detach(),
        }

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,

        # wam
        robot_state: Optional[torch.Tensor] = None,
        robot_state_mask = None,
        robot_action_horizon: Optional[int] = 17,
        wam_infer_mode: Optional[str] = "video",

        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        parallel_mode=None,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        env_steps = os.environ.get("QW_NUM_INFERENCE_STEPS", "").strip()
        if env_steps:
            try:
                num_inference_steps = int(float(env_steps))
            except Exception:
                pass

        # Scheduler
        latent_T = (num_frames - 1) // self.time_division_factor + 1
        if vace_reference_image is not None:
            latent_T += 1

        # 2) latent 
        u = getattr(self.vae, "upsampling_factor", 8)
        latent_H = height // u
        latent_W = width  // u

        # 3) patch size
        ps = getattr(self.dit.video_dit, "patch_size", 4)
        if isinstance(ps, (tuple, list)):
            ph, pw = ps[-2], ps[-1]
        else:
            ph = pw = ps

        dynamic_shift_len = ((latent_H + ph - 1) // ph) * ((latent_W + pw - 1) // pw)

        # 4) 传入 dynamic_shift_len
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            training=False,
            shift=sigma_shift,
            dynamic_shift_len=dynamic_shift_len,
            num_frames=latent_T
        )

        self.scheduler_action.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            training=False,
            shift=sigma_shift,
        )

        batch_size = len(prompt) if isinstance(prompt, (list, tuple)) else 1 

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            "positive": True,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
            "positive": False,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "batch_size": batch_size,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,

        }

        if robot_state is not None:
            # action dit only inputs
            inputs_shared["robot_state"] = robot_state
            inputs_shared["robot_state_mask"] = robot_state_mask
            inputs_shared["robot_action_horizon"] = robot_action_horizon

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        inputs_shared["latents"].to(self.device)

        ar_mode_map = {
            "action_ar": "action",
            "video_ar": "video",
            "video_action_joint_ar": "video_action_joint",
        }
        if wam_infer_mode in ar_mode_map:
            wam_infer_mode = ar_mode_map[wam_infer_mode]
        solver_method = 'euler'
        solver_video = None
        solver_action = None
        
        if wam_infer_mode == "action":
            return self.infer_action(
                inputs_shared,
                inputs_posi,
                inputs_nega,
                switch_DiT_boundary,
                progress_bar_cmd,
                solver_action=solver_action,
                cfg_merge=cfg_merge,
            )
        elif wam_infer_mode == "video":
            return self.infer_video(
                inputs_shared,
                inputs_posi,
                inputs_nega,
                switch_DiT_boundary,
                progress_bar_cmd,
                cfg_scale,
                cfg_merge,
                tiled,
                tile_size,
                tile_stride,
                solver_video=solver_video,
            )
        elif wam_infer_mode == "video_action_joint":
            return self.infer_video_action_joint(
                inputs_shared, 
                inputs_posi, 
                inputs_nega, 
                switch_DiT_boundary, 
                progress_bar_cmd,
                cfg_scale,
                cfg_merge,
                tiled,
                tile_size,
                tile_stride,
                solver_video=solver_video,
                solver_action=solver_action,
            )
        else:
            raise NotImplementedError("Unsupported wam infer mode: {}".format(wam_infer_mode))


    def preprocess_video(self, videos, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        res = []
        min_num_frames = min([len(video) for video in videos])
        w,h = videos[0][0].size
        videos = [
            [np.asarray(img, dtype=np.uint8) for img in video[:min_num_frames]]
            for video in videos
        ]
        try:
            videos = np.stack(
                [np.stack(seq, axis=0) for seq in videos],
                axis=0
            )
        except Exception as e:
            print('video np.stack error, {} || {}'.format(e, videos))
            try:
                print('video np.stack error, ', [vi.shape for vi in videos])
            except:
                pass
            videos = np.zeros((len(videos), 1, h, w, 3), dtype=np.uint8)
        videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3).contiguous()
        # move to cuda
        videos = videos.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device, non_blocking=True)
        # normalize
        videos = videos.mul_((max_value - min_value) / 255).add_(min_value)
        return videos


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: KairosEmbodiedWAMPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "height", "width","batch_size", "num_frames", "seed", "rand_device", "vace_reference_image", "robot_action_horizon"))

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_video, height, width,batch_size, num_frames, seed, rand_device, vace_reference_image, robot_action_horizon):
        if input_video is not None:
            batch_size = len(input_video)
            length = min([(len(video) - 1) // 4 + 1 for video in input_video])
            if length < 1:
                length = 1
        else:
            if batch_size is None:
                batch_size = 1
            else:
                batch_size = batch_size
            length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (batch_size, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)

        if robot_action_horizon is not None:
            shape = [batch_size, robot_action_horizon, pipe.dit.action_dit.action_dim]
            noise_action = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
            return {"noise": noise, "noise_action": noise_action}
        else:
            return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params={"input_image": "input_image"},
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe, prompt, positive, input_image=None) -> dict:
        if positive is not None and not positive:
            input_image = None
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb, attn_mask = pipe.prompter.encode_prompt(prompt, images=input_image, positive=positive, device=pipe.device)
        """
        NOTE: now we move this part outside in train.py, in order to switch it with other conditions
        if self.p_uncond > 0:
            uncond_mask = torch.rand(prompt_emb.shape[0], device=prompt_emb.device) > self.p_uncond
            prompt_emb = torch.where(
                uncond_mask.view(-1, 1, 1), 
                prompt_emb, 
                torch.zeros_like(prompt_emb)
            )
            if attn_mask is not None:
                attn_mask = torch.where(
                    uncond_mask.view(-1, 1),
                    attn_mask,
                    torch.zeros_like(attn_mask)
                )
        """
        return {"context": prompt_emb, "context_mask": attn_mask}


class WanVideoUnit_RobotStateEncodeToContext(PipelineUnit):
    def __init__(self
    ):
        super().__init__(
            seperate_cfg=True,
            input_params={"robot_state": "robot_state" , 'robot_state_mask': 'robot_state_mask'},
            input_params_posi={"positive": "positive", "context": "context", "context_mask": "context_mask"},
            input_params_nega={"positive": "positive", "context": "context", "context_mask": "context_mask"},
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, robot_state, robot_state_mask, positive, context, context_mask):

        if robot_state is None:
            return {}
        
        if not (hasattr(pipe.dit, 'action_dit') and hasattr(pipe.dit.action_dit, 'proprio_encoder')):
            print('pipe.dit.action_dit.proprio_encoder not found')
            return {}

        if pipe.dit.action_dit.proprio_encoder.training:
            with torch.enable_grad():
                return process_robot_state_to_context(pipe, robot_state, robot_state_mask, positive, context, context_mask)
        else:
            return process_robot_state_to_context(pipe, robot_state, robot_state_mask, positive, context, context_mask)


def process_robot_state_to_context(pipe, robot_state, robot_state_mask, positive, context, context_mask):
    
    assert context is not None
    context_raw_shape = context.shape

    if robot_state is None:
        return {}

    if robot_state_mask is None:
        robot_state_mask = torch.ones_like(robot_state, dtype=torch.bool, device=context.device)
    assert robot_state.shape[0] == robot_state_mask.shape[0]

    robot_state = robot_state.to(dtype=context.dtype, device=context.device)
    robot_state_mask = robot_state_mask.to(device=context.device)

    if robot_state.dim() == 2:
        robot_state = robot_state.unsqueeze(1) 
        robot_state_mask = robot_state_mask.unsqueeze(1)
    
    robot_state = robot_state * robot_state_mask
    state_token_mask = (robot_state_mask.sum(dim=-1) > 0)  
    proprio_token = pipe.dit.action_dit.proprio_encoder(
            robot_state.to(device=context.device, dtype=context.dtype)
            ).to(dtype=context.dtype) 
    context = torch.cat([context, proprio_token], dim=1)
    gen_mask = (robot_state_mask is not None) or (context_mask is not None)
    if gen_mask:
        if context_mask is None:
            context_mask = torch.ones((context_raw_shape[0], context_raw_shape[1]), dtype=torch.bool, device=context.device)
        if robot_state_mask is None:
            proprio_mask = torch.ones((proprio_token.shape[0], proprio_token.shape[1]), dtype=torch.bool, device=context_mask.device)
        else:
            proprio_mask = state_token_mask.to(context.device)
        context_mask = torch.cat([context_mask, proprio_mask], dim=1)
        assert context_mask.shape[1] == context.shape[1]
    return {"context": context, "context_mask": context_mask}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """

    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.video_dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """

    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        fuse_vae_embedding_in_latents = pipe.dit.video_dit.fuse_vae_embedding_in_latents

        if input_image is None or not fuse_vae_embedding_in_latents:
            return {}
        if not isinstance(input_image, (list, tuple)):
            input_image = [input_image]
        pipe.load_models_to_device(self.onload_model_names)
        images = []
        for image in input_image:
            images.append(pipe.preprocess_image(image.resize((width, height))).transpose(0, 1))
        z = pipe.vae.encode(images, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)

        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: KairosEmbodiedWAMPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: KairosEmbodiedWAMPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: KairosEmbodiedWAMPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: KairosEmbodiedWAMPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = [
            "context",
            "context_mask",
            "clip_feature",
            "y",
            "reference_latents",
        ]

    def process(self, pipe: KairosEmbodiedWAMPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )

    def process_audio(self, pipe: KairosEmbodiedWAMPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: KairosEmbodiedWAMPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: KairosEmbodiedWAMPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: KairosEmbodiedWAMPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio"), inputs_shared.pop("audio_embeds"), inputs_shared.get("audio_sample_rate")
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video"), inputs_shared.pop("s2v_pose_latents"), inputs_shared.pop("motion_video")

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: KairosEmbodiedWAMPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: KairosEmbodiedWAMPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x

        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype)
                for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value

def ddp_or_bool(value: bool, device: torch.device) -> bool:
    """
    DDP-safe logical OR:
    if any rank evaluates to True, all ranks will return True.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value

    t = torch.tensor(
        int(value),
        device=device,
        dtype=torch.int32,
    )
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return bool(t.item())


def model_fn_wan_video(
    dit,
    motion_controller = None,
    vace = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    context_mask: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    gradient_checkpointing_level: str = 'block',
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    first_frame_latents: Optional[torch.Tensor] = None,

    # *************************************************
    # action
    action_latents: Optional[torch.Tensor] = None,
    timestep_action: Optional[torch.Tensor] = None,

    **kwargs,
):

    assert len(latents.shape) == 5
    # [B, C, T, H, W]
    b,c,t,h,w = latents.shape
    video_seq_len = t * h * w // dit.video_dit.patch_size[1] // dit.video_dit.patch_size[2]
    video_tokens_per_frame = h * w // dit.video_dit.patch_size[1] // dit.video_dit.patch_size[2]
    if first_frame_latents is not None:
        num_history_frames = first_frame_latents.shape[2]
    else:
        num_history_frames = 0

    if timestep_action is None:
        # video only forward

        attention_mask_d = build_mot_attention_mask_info_dict_wrapper(
            video_seq_len=video_seq_len,
            action_seq_len=0,
            video_tokens_per_frame=video_tokens_per_frame,
            num_history_frames=num_history_frames,
            device=latents.device,
            mask_types=['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
            all_atten_history = dit.video_dit.all_atten_history,
            atten_modes = dit.video_dit.atten_modes,
            window_size=dit.video_dit.window_size,
            dilated_length=dit.video_dit.dilated_length,
            flex_block_size=dit.video_dit.flex_block_size,
            attn_method=dit.video_dit.attn_method,
            restrict_history_query_to_history=dit.video_dit.restrict_history_query_to_history,
        )

        pred_video = mot_infer_pure_video_dit_once(
            video_dit=dit.video_dit,
            latents=latents,
            first_frame_latents=first_frame_latents,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            timestep = timestep,
            context=context,
            context_mask=context_mask,
            attention_mask_d=attention_mask_d,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            gradient_checkpointing_level='block',
        )

        pred_action = None
    
    else:
        # joint forward

        action_seq_len = action_latents.shape[1]
        attention_mask_d = build_mot_attention_mask_info_dict_wrapper(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_history_frames=num_history_frames,
            device=latents.device,
            mask_types=['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
            all_atten_history = dit.video_dit.all_atten_history,
            atten_modes = dit.video_dit.atten_modes,
            window_size=dit.video_dit.window_size,
            dilated_length=dit.video_dit.dilated_length,
            flex_block_size=dit.video_dit.flex_block_size,
            attn_method=dit.video_dit.attn_method,
            restrict_history_query_to_history=dit.video_dit.restrict_history_query_to_history,
        )

        pred_video, pred_action = mot_infer_joint_video_action_dit_once(
            video_dit=dit.video_dit,
            action_dit=dit.action_dit,
            # *************************************************
            # shared inputs
            context = context,
            context_mask = context_mask,
            attention_mask_d = attention_mask_d,
            # *************************************************
            # video inputs
            latents = latents,
            timestep = timestep,
            fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents,
            first_frame_latents = first_frame_latents,
            # *************************************************
            # action inputs
            action_latents = action_latents,
            timestep_action = timestep_action,

            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            gradient_checkpointing_level='block',
        )
    return pred_video, pred_action
