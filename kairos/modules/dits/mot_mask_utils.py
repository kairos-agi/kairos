from collections import defaultdict
import sys
import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open
from einops import rearrange



# *************************************************************************************
# mot code

# ***************************
# torch flex mot-mask || start


from functools import lru_cache

import torch
from torch.nn.attention.flex_attention import create_block_mask


@torch.no_grad()
def _build_mot_mask_mod(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_history_frames: int,
    video_atten_mode: str = "FA",
    window_size: int = 3,
    dilated_length: int = 4,
    all_atten_history: bool = False,
    restrict_history_query_to_history: bool = True,
):
    assert video_atten_mode in ["FA", "SWA", "DSWA"]
    assert video_seq_len % video_tokens_per_frame == 0

    total_seq_len = video_seq_len + action_seq_len
    num_frames = video_seq_len // video_tokens_per_frame
    history_frame_tokens = min(
        video_tokens_per_frame * num_history_frames, video_seq_len
    )

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_video = q_idx < video_seq_len
        kv_is_video = kv_idx < video_seq_len

        q_is_action = ~q_is_video
        kv_is_action = ~kv_is_video

        has_history = history_frame_tokens > 0
        q_is_history_video = (q_idx < history_frame_tokens) if has_history else torch.zeros_like(q_idx, dtype=torch.bool)
        kv_is_history_video = (kv_idx < history_frame_tokens) if has_history else torch.zeros_like(kv_idx, dtype=torch.bool)

        # -------------------------
        # case 1: video -> video
        # -------------------------
        vv = q_is_video & kv_is_video

        if video_atten_mode == "FA":
            vv_visible = torch.ones_like(vv, dtype=torch.bool)

        elif video_atten_mode == "SWA":
            q_frame = q_idx // video_tokens_per_frame
            kv_frame = kv_idx // video_tokens_per_frame
            vv_visible = torch.abs(q_frame - kv_frame) < window_size

        elif video_atten_mode == "DSWA":
            q_frame = q_idx // video_tokens_per_frame
            kv_frame = kv_idx // video_tokens_per_frame
            frame_diff = kv_frame - q_frame
            vv_visible = (
                (frame_diff % dilated_length == 0)
                & (torch.abs(frame_diff // dilated_length) < window_size)
            )
        else:
            raise ValueError(f"Invalid video attention mode: {video_atten_mode}")

        if restrict_history_query_to_history:
            vv_visible = vv_visible & ~(q_is_history_video & ~kv_is_history_video)

        if all_atten_history:
            vv_visible = vv_visible | (q_is_video & kv_is_history_video)

        # -------------------------
        # case 2: action -> action
        # -------------------------
        aa_visible = q_is_action & kv_is_action

        # -------------------------
        # case 3: action -> history video tokens
        # -------------------------
        av_visible = q_is_action & kv_is_video & kv_is_history_video

        # -------------------------
        # case 4: video -> action = False
        # -------------------------
        visible = (vv & vv_visible) | aa_visible | av_visible
        return visible

    return mask_mod, total_seq_len, history_frame_tokens


@torch.no_grad()
def _build_region_mask_mod(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_history_frames: int,
    video_atten_mode: str = "FA",
    window_size: int = 3,
    dilated_length: int = 4,
    all_atten_history: bool = False,
    restrict_history_query_to_history: bool = True,
    mask_type: str = "video_action_mixed",
):
    full_mask_mod, total_seq_len, _ = _build_mot_mask_mod(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        num_history_frames=num_history_frames,
        video_atten_mode=video_atten_mode,
        window_size=window_size,
        dilated_length=dilated_length,
        all_atten_history=all_atten_history,
        restrict_history_query_to_history=restrict_history_query_to_history,
    )

    assert mask_type in [
        "video_action_mixed",
        "video_only",
        "action_only",
        "action_with_video_cache",
    ]

    if mask_type == "video_action_mixed":
        q_len = total_seq_len
        kv_len = total_seq_len

        def region_mask_mod(b, h, q_idx, kv_idx):
            return full_mask_mod(b, h, q_idx, kv_idx)

    elif mask_type == "video_only":
        q_len = video_seq_len
        kv_len = video_seq_len

        def region_mask_mod(b, h, q_idx, kv_idx):
            return full_mask_mod(b, h, q_idx, kv_idx)

    elif mask_type == "action_only":
        q_len = action_seq_len
        kv_len = action_seq_len

        def region_mask_mod(b, h, q_idx, kv_idx):
            q_global = q_idx + video_seq_len
            kv_global = kv_idx + video_seq_len
            return full_mask_mod(b, h, q_global, kv_global)

    elif mask_type == "action_with_video_cache":
        q_len = action_seq_len
        kv_len = total_seq_len

        def region_mask_mod(b, h, q_idx, kv_idx):
            q_global = q_idx + video_seq_len
            kv_global = kv_idx
            return full_mask_mod(b, h, q_global, kv_global)

    else:
        raise ValueError(f"Invalid mask_type: {mask_type}")

    return region_mask_mod, q_len, kv_len


@lru_cache(maxsize=128)
def _cached_block_mask_by_type(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_history_frames: int,
    video_atten_mode: str,
    window_size: int,
    dilated_length: int,
    all_atten_history: bool,
    restrict_history_query_to_history: bool,
    mask_type: str,
    block_size: int,
    device_str: str,
):
    mask_mod, q_len, kv_len = _build_region_mask_mod(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        num_history_frames=num_history_frames,
        video_atten_mode=video_atten_mode,
        window_size=window_size,
        dilated_length=dilated_length,
        all_atten_history=all_atten_history,
        restrict_history_query_to_history=restrict_history_query_to_history,
        mask_type=mask_type,
    )

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device_str,
        BLOCK_SIZE=block_size,
    )


@torch.no_grad()
def build_mot_flex_block_mask(
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    num_history_frames: int,
    device: str,
    video_atten_mode: str = "FA",
    window_size: int = 3,
    dilated_length: int = 4,
    all_atten_history: bool = False,
    restrict_history_query_to_history: bool = True,
    block_size: int = 128,
    mask_type: str = "video_action_mixed",
):
    return _cached_block_mask_by_type(
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=video_tokens_per_frame,
        num_history_frames=num_history_frames,
        video_atten_mode=video_atten_mode,
        window_size=window_size,
        dilated_length=dilated_length,
        all_atten_history=all_atten_history,
        restrict_history_query_to_history=restrict_history_query_to_history,
        mask_type=mask_type,
        block_size=block_size,
        device_str=device,
    )


@torch.no_grad()
def build_mot_flex_attention_mask_dict(
    video_seq_len,
    action_seq_len,
    video_tokens_per_frame,
    num_history_frames,
    device,
    all_atten_history = False,
    atten_modes = ['FA','SWA', 'DSWA'],
    mask_types = ['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
    window_size = 3,
    dilated_length = 4,
    flex_block_size = 128,
    restrict_history_query_to_history: bool = True,
):

    attention_mask_d = dict()

    if action_seq_len == 0:
        mask_types = ['video_only']
    elif video_seq_len == 0:
        mask_types = ['action_only']
    
    for mask_type in mask_types:

        if mask_type == 'action_only':
            _atten_modes = ['FA']
        else:
            _atten_modes = atten_modes

        for atten_mode in _atten_modes:
            key_name = f"{mask_type}@{atten_mode}"
            
            flex_attn_mask = build_mot_flex_block_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=video_tokens_per_frame,
                num_history_frames=num_history_frames,
                device=device,
                video_atten_mode=atten_mode,
                window_size=window_size,
                dilated_length=dilated_length,
                all_atten_history=all_atten_history,
                restrict_history_query_to_history=restrict_history_query_to_history,
                mask_type=mask_type,
                block_size=flex_block_size,
            )

            extra_attn_info = dict() # when using flex attention, extra_attn_info is empty, just placeholder
            attention_mask_d[key_name] = [flex_attn_mask, extra_attn_info]

    return attention_mask_d


# torch flex mot-mask || end
# ****************************
# flash attention mot-mask-info || start

def build_mot_flash_attention_mask_info(
    video_seq_len,
    action_seq_len,
    video_tokens_per_frame,
    num_history_frames,
    device,
    all_atten_history = False,
    atten_modes = ['FA','SWA', 'DSWA'],
    mask_types = ['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
    window_size = 3,
    dilated_length = 4,
    flex_block_size = 128,
):

    attention_mask_d = dict()

    if action_seq_len == 0:
        mask_types = ['video_only']
    elif video_seq_len == 0:
        mask_types = ['action_only']
    
    for mask_type in mask_types:

        if mask_type == 'action_only':
            _atten_modes = ['FA']
        else:
            _atten_modes = atten_modes

        for atten_mode in _atten_modes:
            key_name = f"{mask_type}@{atten_mode}"

            if atten_mode == 'FA':
                real_window_size = -1
                real_dilated_length = 1
            elif atten_mode == 'SWA':
                real_window_size =window_size
                real_dilated_length = 1
            elif atten_mode == 'DSWA':
                real_window_size = window_size
                real_dilated_length = dilated_length
            else:
                raise ValueError(f"Invalid atten_mode: {atten_mode}")

            extra_attn_info = dict(
                video_tokens_per_frame=video_tokens_per_frame,
                num_history_frames=num_history_frames,
                window_size=real_window_size,
                dilated_length=real_dilated_length,
            )

            if atten_mode == 'DSWA':
                extra_attn_info['dswa_seq_len'] = video_seq_len
            else:
                extra_attn_info['dswa_seq_len'] = 1

            attn_mask = None

            attention_mask_d[key_name] = [attn_mask, extra_attn_info]


    return attention_mask_d
# flash attention mot-mask-info || end
# ****************************



def build_mot_attention_mask_info_dict_wrapper(
    video_seq_len,
    action_seq_len,
    video_tokens_per_frame,
    num_history_frames,
    device,
    all_atten_history = False,
    atten_modes = ['FA','SWA', 'DSWA'],
    mask_types = ['video_action_mixed', 'video_only', 'action_only', 'action_with_video_cache'],
    window_size = 3,
    dilated_length = 4,
    flex_block_size = 128,
    attn_method='flash',
    restrict_history_query_to_history: bool = True,
):

    if attn_method == 'flex':
        attn_mask_d = build_mot_flex_attention_mask_dict(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            all_atten_history=all_atten_history,
            device=device,
            num_history_frames=num_history_frames,
            atten_modes=atten_modes,
            mask_types=mask_types,
            window_size=window_size,
            dilated_length=dilated_length,
            flex_block_size=flex_block_size,
            restrict_history_query_to_history=restrict_history_query_to_history,
        )
    elif attn_method == 'flash':
        attn_mask_d = build_mot_flash_attention_mask_info(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            num_history_frames=num_history_frames,
            device=device,
            all_atten_history=all_atten_history,
            atten_modes=atten_modes,
            mask_types=mask_types,
            window_size=window_size,
            dilated_length=dilated_length,
            flex_block_size=flex_block_size,
        )
        
    else:
        raise ValueError(f"Invalid attn_method: {attn_method}")

    return attn_mask_d



