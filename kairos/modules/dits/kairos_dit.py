import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from transformers.activations import ACT2CLS
from apex.normalization.fused_layer_norm import FusedRMSNorm

FLASH_ATTN_2_AVAILABLE = False
FLASH_ATTN_3_AVAILABLE = False
SAGE_ATTN_AVAILABLE = False

try:
    import flash_attn_interface
    # FLASH_ATTN_3_AVAILABLE = True
    # NOTE: forcing disable flash_attn3 for a800
    FLASH_ATTN_3_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

IS_CUDA = torch.cuda.is_available()
if not IS_CUDA:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

from kairos.modules.utils import FLAGS_KAIROS_CUDA_SM

SUPPORTED_ARCHS = {80, 89, 120, 121}
if FLAGS_KAIROS_CUDA_SM in SUPPORTED_ARCHS:
    try:
        from sageattention import sag_attention_with_window,sageattn
        SAGE_ATTN_AVAILABLE = True
    except ModuleNotFoundError:
        SAGE_ATTN_AVAILABLE = False
try:
    from fla.layers import GatedDeltaNet
except ModuleNotFoundError:
    pass

from kairos.apis.builder import DITS
from torch.distributed import ProcessGroup, get_process_group_ranks
from kairos.third_party.fla.layers.gated_deltanet_with_tp import GatedDeltaNet as GatedDeltaNetWithTP
from kairos.third_party.fla.models.utils import Cache as fla_Cahce
import torch.distributed as dist
from kairos.modules.utils import parallel_state, FLAGS_KAIROS_IS_METAX
from kairos.modules.utils.tp_utils import build_tp_chunk_list, all2all_seq_to_head, all2all_head_to_seq, _gather_input_sp, _distribute_input_sp

#================================================================
#====================rope_apply_for3d_triton当前只适用于沐曦========
#================================================================
import triton
import triton.language as tl
from fla.utils import autotune_cache_kwargs

def rope_apply_for3d_triton(x, num_frames, freqs, num_heads):
    assert x.is_contiguous(), "x 必须是 contiguous 的 [B,L,D]"
    B, T, D = x.shape
    H = num_heads
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads) # [B,T,H,HD] ([1,720000,20,128]) bf16
    HD = x.shape[-1]
    C = HD // 2 # 64
    # freqs: [L, 1, HD/2] complex
    assert freqs.shape[0] == T and freqs.shape[1] == 1
    assert HD % freqs.shape[-1] == 0
    assert freqs.shape[-1] == C
    freqs_scalar = torch.view_as_real(freqs)
    assert x.is_contiguous()
    assert freqs_scalar.is_contiguous()

    # 输出 buffer，同 layout
    out = torch.empty_like(x) # [B,L,H,D] ([1,720000,20,128]) bf16

    def grid(meta):
        BT = meta['BT']
        return (triton.cdiv(T, BT), B * H)
    rope_apply_for_3d_kernel[grid](
        x, freqs_scalar, out,
        B, T, H, C, HD,
    )
    out = out.reshape(B, T, D)
    return out

@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [8]
        for BT in [32]
    ],
    key=['B', 'T'],
    use_cuda_graph=False,
    **autotune_cache_kwargs
)
@triton.jit
def rope_apply_for_3d_kernel(
    x_bf16,
    freqs_f64,
    out_bf16,
    B,
    T,
    H:  tl.constexpr,
    C:  tl.constexpr,
    HD: tl.constexpr,
    BT: tl.constexpr,
):
    pid_t  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H
    if b >= B:
        return

    t_start = pid_t * BT

    # 复数通道 index: 0..C-1
    offs_c = tl.arange(0, C)  # [C]

    for dt in range(0, BT):
        t = t_start + dt
        mask_t = t < T         # 当前这一行是否有效（标量 bool）

        # 如果这一 dt 已经越界了，不 return，只是把 load/store 全部 mask 掉
        # x[b, t, h, d]，d = 0..HD-1
        base_x = ((b * T + t) * H + h) * HD

        idx_real = base_x + 2 * offs_c       # d = 2c
        idx_imag = idx_real + 1              # d = 2c + 1

        # 每个通道都复用同一个 time-mask
        mask_elem = mask_t                   # 形状 [C]，广播到 [C]

        # -------- x: bf16 → float64，拆成实部/虚部 --------
        x_real_bf16 = tl.load(
            x_bf16 + idx_real,
            mask=mask_elem,
            other=0.0,
        )                                    # [C]
        x_imag_bf16 = tl.load(
            x_bf16 + idx_imag,
            mask=mask_elem,
            other=0.0,
        )                                    # [C]
        x_real_f64 = x_real_bf16.to(tl.float64)
        x_imag_f64 = x_imag_bf16.to(tl.float64)

        # -------- freqs: float64 视图 [T, 2*C] --------
        # 假设 freqs_f64 按 [t, 2*C] 摊平成一维：
        # real 在偶数位，imag 在奇数位
        base_f = t * (2 * C)                 # 对应 freqs[t, 0]
        idx_freq_real = base_f + 2 * offs_c
        idx_freq_imag = idx_freq_real + 1

        freq_real_f64 = tl.load(
            freqs_f64 + idx_freq_real,
            mask=mask_elem,
            other=0.0,
        )
        freq_imag_f64 = tl.load(
            freqs_f64 + idx_freq_imag,
            mask=mask_elem,
            other=0.0,
        )

        # -------- 复数乘 (fp64) --------
        # (x_r + i x_i) * (f_r + i f_i)
        y_real_f64 = x_real_f64 * freq_real_f64 - x_imag_f64 * freq_imag_f64
        y_imag_f64 = x_real_f64 * freq_imag_f64 + x_imag_f64 * freq_real_f64

        # 先 f64 -> f32
        y_real_f32 = y_real_f64.to(tl.float32)
        y_imag_f32 = y_imag_f64.to(tl.float32)

        # 再 f32 -> bf16
        y_real_bf16 = y_real_f32.to(tl.bfloat16)
        y_imag_bf16 = y_imag_f32.to(tl.bfloat16)

        tl.store(
            out_bf16 + idx_real,
            y_real_bf16,
            mask=mask_elem,
        )
        tl.store(
            out_bf16 + idx_imag,
            y_imag_bf16,
            mask=mask_elem,
        )

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, 
                    compatibility_mode=False, attn_mask=None, window_size=(-1, -1), return_attn_probs=False):
    
    # ***************************************
    # debug code
    # if DEBUG_CLOSE_FLASH_ATTN:
    #     compatibility_mode = True
    # debug code
    # ***************************************
    if compatibility_mode or attn_mask is not None:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s n d")
    elif FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_func(
            q, k, v, window_size=window_size,
            return_attn_probs=return_attn_probs
        )
        if return_attn_probs:
            x, probs = x[0], x[1]
            return x, probs
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_func(
            q, k, v, window_size=window_size,
            return_attn_probs=return_attn_probs
        )
        if return_attn_probs:
            x, probs = x[0], x[1]
            return x, probs
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s n d -> b n s d")
        k = rearrange(k, "b s n d -> b n s d")
        v = rearrange(v, "b s n d -> b n s d")
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s n d")
    else:
        raise RuntimeError("do not use pytorch attention")
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    x.mul_(1 + scale)
    x.add_(shift)
    return x


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float16).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

def rope_apply_for3d(x, num_frames, freqs, num_heads):
    B, L, D = x.shape
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)

    x_out = torch.view_as_complex(x.to(torch.float16).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)

    x_out = x_out.reshape(B, L, D)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attn_mask=None, window_size=(-1, -1)):
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attn_mask=attn_mask, window_size=window_size)
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        return x
class SagAttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pv_accum_dtype = "fp16"
        self.smooth_v=True
        if FLAGS_KAIROS_CUDA_SM in [89, 120, 121]:
            self.pv_accum_dtype = "fp32+fp16"
            self.smooth_v = True
        
    def forward(self, q, k, v, attn_mask=None, window_size=(-1, -1)):
        b, s, _ = q.shape
        n = self.num_heads
        d = q.shape[-1] // n

        # 一步到位：b s (n d) -> b n s d
        q = q.view(b, s, n, d).transpose(1, 2)
        k = k.view(b, s, n, d).transpose(1, 2)
        v = v.view(b, s, n, d).transpose(1, 2)
        x = sag_attention_with_window(q, k, v,window_size=window_size,qk_quant_gran="per_warp", pv_accum_dtype=self.pv_accum_dtype,smooth_v=self.smooth_v)
        x = x.transpose(1, 2).contiguous().view(b, s, n * d)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, dilated_length=1, window_size=3, attend_k0=False, backend = "flashattention"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dilated_length = dilated_length
        self.window_size = window_size
        self.attend_k0 = attend_k0
        self.backend = backend

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = FusedRMSNorm(dim, eps=eps)
        self.norm_k = FusedRMSNorm(dim, eps=eps)

        if SAGE_ATTN_AVAILABLE:
            self.attn = SagAttentionModule(self.num_heads)
        else:
            self.attn = AttentionModule(self.num_heads)
 
    def extra_repr(self):
        return f'dilated_length={self.dilated_length}, window_size={self.window_size}'
    
    def forward(self, x, f, freqs, L=1):
        dilated_length = self.dilated_length
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        use_dilated = dilated_length > 1 and x.shape[1] // (dilated_length * L) > 1
        

        if use_dilated:
            assert x.shape[1] % L == 0, "L should equal to the num of tokens per frame"
            pad_len = dilated_length * L - x.shape[1] % (dilated_length * L)
            if pad_len != 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
            q = rearrange(q, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            k = rearrange(k, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            v = rearrange(v, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
        x = self.attn(q, k, v, window_size=(L*self.window_size, L*self.window_size))

        out = self.o(x)
        if use_dilated:
            out = rearrange(out, "(b d) (n l) c -> b (n d l) c", l=L, d=dilated_length)
            if pad_len != 0:
                out = out[:, :-pad_len]
        return out

def tensor_parallel_rms_norm(x, norm, tp_chunk_list):
    tp_rank = parallel_state.get_context_parallel_rank()
    tp_group = parallel_state.get_context_parallel_group()

    # full dim from RMSNorm
    full_dim = norm.weight.shape[0]
    total_heads = sum(tp_chunk_list)

    # head_dim 必须来自 full_dim / total_heads
    assert full_dim % total_heads == 0, \
        f"RMSNorm dim {full_dim} not divisible by total_heads {total_heads}"

    head_dim = full_dim // total_heads
    local_heads = tp_chunk_list[tp_rank]
    local_dim = local_heads * head_dim

    start = sum(tp_chunk_list[:tp_rank]) * head_dim
    end = start + local_dim

    assert x.shape[-1] == local_dim, \
        f"RMSNorm input dim mismatch: x={x.shape[-1]}, expect {local_dim}"

    weight = norm.weight[start:end]

    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

    dist.all_reduce(variance, op=dist.ReduceOp.AVG, group=tp_group)

    out = x_fp32 * torch.rsqrt(variance + norm.eps) * weight
    return out.to(x.dtype)

def _distribute_freqs_sp(freqs, context_group):
    context_group_world_size = dist.get_world_size(group=context_group)
    context_group_rank = dist.get_rank(group=context_group)
    if context_group_world_size <= 1:
        return freqs

    N, B, D = freqs.shape
    assert N % context_group_world_size == 0, 'cannot split tensor {} vs {}'.format(freqs.shape, context_group_world_size)
    local_dim = N // context_group_world_size
    start_idx = context_group_rank * local_dim
    end_idx = start_idx + local_dim if context_group_rank != context_group_world_size - 1 else N

    local_freqs = freqs[start_idx:end_idx, :, :].contiguous()
    return local_freqs

class SelfAttentionTP(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, dilated_length=1, window_size=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.tp_group = parallel_state.get_context_parallel_group()
        self.tp_size = parallel_state.get_context_parallel_world_size()

        if self.num_heads % self.tp_size != 0:
            self.tp_chunk_list = build_tp_chunk_list(self.num_heads, self.tp_size)
            num_heads_local = self.tp_chunk_list[
                parallel_state.get_context_parallel_rank()
            ]
        else:
            self.tp_chunk_list = None
            num_heads_local = self.num_heads // self.tp_size

        self.head_dim = dim // self.num_heads
        assert dim % self.num_heads == 0, "dim must be divisible by num_heads"

        self.dilated_length = dilated_length
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = FusedRMSNorm(dim, eps=eps)
        self.norm_k = FusedRMSNorm(dim, eps=eps)

        if SAGE_ATTN_AVAILABLE:
            self.attn = SagAttentionModule(num_heads_local)
        else:
            self.attn = AttentionModule(num_heads_local)

        self._debug_saved_attn_out = False

    def extra_repr(self):
        return f'dilated_length={self.dilated_length}, window_size={self.window_size}, tp_size={self.tp_size}'


    def forward(self, x, f, freqs, L=1):
        dilated_length = self.dilated_length
    
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if FLAGS_KAIROS_IS_METAX:
            q = rope_apply_for3d_triton(q, f, freqs, self.num_heads)
            k = rope_apply_for3d_triton(k, f, freqs, self.num_heads)
        else:
            q = rope_apply_for3d(q, f, freqs, self.num_heads)
            k = rope_apply_for3d(k, f, freqs, self.num_heads)

        q = all2all_seq_to_head(q, self.tp_group, self.tp_chunk_list, self.head_dim)
        k = all2all_seq_to_head(k, self.tp_group, self.tp_chunk_list, self.head_dim)
        v = all2all_seq_to_head(v, self.tp_group, self.tp_chunk_list, self.head_dim)

        seq_local = x.shape[1]
        seq_global = seq_local * self.tp_size
        use_dilated = dilated_length > 1 and seq_global // (dilated_length * L) > 1
    
        if use_dilated:
            # Sequence length is generally divisible; cases where it is not divisible are not supported.

            assert seq_global % L == 0, "L should equal to the num of tokens per frame"
            pad_len = dilated_length * L - seq_global % (dilated_length * L)

            if pad_len != 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))

            q = rearrange(q, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            k = rearrange(k, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)
            v = rearrange(v, "b (n d l) c -> (b d) (n l) c", l=L, d=dilated_length)

        x = self.attn(q, k, v, window_size=(L*self.window_size, L*self.window_size))

        if use_dilated:
            x = rearrange(x, "(b d) (n l) c -> b (n d l) c", l=L, d=dilated_length)
            if pad_len != 0:
                x = x[:, :-pad_len]
        x = all2all_head_to_seq(x, self.tp_group, self.tp_chunk_list, self.head_dim)

        out = self.o(x)

        return out

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = FusedRMSNorm(dim, eps=eps)
        self.norm_k = FusedRMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = FusedRMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor, attn_mask=None):

        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        if attn_mask is not None:
            B, L, S = x.shape[0], x.shape[1], ctx.shape[1]
            attn_mask = attn_mask.view(B, 1, 1, S).expand(B, 1, L, S)

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v, attn_mask=attn_mask)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

def _broadcast_from_owner(x_chunk, owner_rank, chunk_shape, dtype, device, group):
    rank = dist.get_rank(group)
    if rank != owner_rank:
        x_chunk = torch.empty(chunk_shape, device=device, dtype=dtype)
    dist.broadcast(x_chunk, src=owner_rank, group=group)
    return x_chunk


def _get_owner_chunk_info(chunk_id, chunk_size, local_seq_len):
    global_start = chunk_id * chunk_size
    owner_rank = global_start // local_seq_len
    owner_local_start = global_start % local_seq_len
    owner_local_end = owner_local_start + chunk_size
    return owner_rank, owner_local_start, owner_local_end

def _all_gather_seq_chunk(o_seq_part, group, q_len=None):
    world_size = dist.get_world_size(group)
    gather_list = [torch.empty_like(o_seq_part) for _ in range(world_size)]
    dist.all_gather(gather_list, o_seq_part.contiguous(), group=group)
    out = torch.cat(gather_list, dim=1).contiguous()
    if q_len is not None and out.shape[1] > q_len:
        out = out[:, :q_len, :]
    return out

class DiTBlock(nn.Module):
    def __init__(self, 
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int, 
        eps: float = 1e-6,
        use_linear_attn = True, 
        dilated_length=1, 
        window_size=3,
        # *************************
        # seq parallel params
        block_idx=-1,
        gateddeltanet_layer_idx = -1,
        is_first_block=False,
        is_last_block=False,
        use_seq_parallel=False,
        use_tp_in_getaeddeltanet=False,
        use_tp_in_self_attn=False,
        attend_k0=False
        ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.use_linear_attn = use_linear_attn

        self.block_idx=block_idx
        self.is_first_block=is_first_block
        self.is_last_block=is_last_block
        self.use_seq_parallel=use_seq_parallel
        self.use_tp_in_getaeddeltanet = use_tp_in_getaeddeltanet
        self.use_tp_in_self_attn = use_tp_in_self_attn

        dist_on = dist.is_available() and dist.is_initialized()
        self.world = dist.get_world_size() if dist_on else 1

        if self.world > 1 and (self.use_seq_parallel or use_tp_in_getaeddeltanet or use_tp_in_self_attn):
            self.context_group_rank = parallel_state.get_context_parallel_rank()
            self.context_group_size = parallel_state.get_context_parallel_world_size()
            self.context_group = parallel_state.get_context_parallel_group()
        else:
            self.context_group_rank = 0  # 单卡/单机非分布式场景默认rank为0
            self.context_group_size = 1  # 单卡/单机非分布式场景默认world_size为1
            self.context_group = None

        if is_first_block:
            print(f'{self.__class__.__name__} use_seq_parallel: {use_seq_parallel} context_group_size: {self.context_group_size}')

        if self.use_linear_attn:
            assert gateddeltanet_layer_idx >= 0

            # getaeddeltanet用tp并行
            if self.use_tp_in_getaeddeltanet and self.world > 1:
                self.gated_delta = GatedDeltaNetWithTP(
                    hidden_size=dim,
                    num_heads=num_heads,
                    mode="chunk",
                    use_gate=True,
                    norm_eps=eps,
                    tp_num_splits=self.context_group_size,
                    tp_group=self.context_group,
                    layer_idx=gateddeltanet_layer_idx,
                )
            else:
                self.gated_delta = GatedDeltaNet(hidden_size=dim,
                                                num_heads=num_heads,
                                                mode='chunk',
                                                use_gate=True,
                                                norm_eps=eps,
                                                layer_idx=gateddeltanet_layer_idx
                                                )
        else:
            assert gateddeltanet_layer_idx == -1
            if self.use_tp_in_self_attn and self.world > 1:
                # self_attention用tp并行
                if attend_k0:
                    raise NotImplementedError('SelfAttentionTP not support attend_k0 yet!!!')
                self.self_attn = SelfAttentionTP(dim, num_heads, eps, dilated_length=dilated_length, window_size=window_size)
            else:
                self.self_attn = SelfAttention(dim, num_heads, eps, dilated_length=dilated_length, window_size=window_size, attend_k0=attend_k0)

        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)

        self.self_attn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.cross_attn_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            ACT2CLS['silu'](),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    def forward(self, x, context, t_mod, freqs, grid_size, context_mask=None):
        (f, h, w) = grid_size
        B, _, D = x.shape
        L = h * w

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        # scale & gate
        mod_dtype = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        t_mod_chunks = t_mod.chunk(6, dim=chunk_dim)
        mod_chunks = mod_dtype.chunk(6, dim=1)

        # for delayed computation
        def get_mod_chunk(idx, sp_slice=None):
            t_c = t_mod_chunks[idx]
            m_c = mod_chunks[idx]

            if has_seq:
                t_c = t_c.squeeze(2)
                m_c = m_c.squeeze(1)
                if sp_slice is not None:
                    start_idx, end_idx = sp_slice
                    t_c = t_c[:, start_idx:end_idx, :]

            return t_c + m_c

        scale_msa = get_mod_chunk(0)
        shift_msa = get_mod_chunk(1)
        gate_msa = get_mod_chunk(2)
        # self-attention
        input_x_local = modulate(self.self_attn_norm(x), shift_msa, scale_msa)
        del scale_msa, shift_msa
        world_size = self.world
        if self.use_linear_attn:
            rank = getattr(self, "context_group_rank", 0)

            if world_size > 1 and (not dist.is_available() or not dist.is_initialized()):
                raise RuntimeError("Distributed must be initialized for multi-rank linear_attn path.")

            B, local_seq_len, hidden_dim = input_x_local.shape

            if world_size > 1:
                chunk_size = local_seq_len
            else:
                assert local_seq_len >= 8 and local_seq_len % 8 == 0, \
                    f"single-card path requires local_seq_len divisible by 8, got {local_seq_len}"
                chunk_size = local_seq_len // 8

            assert local_seq_len % chunk_size == 0, \
                f"chunk_size={chunk_size} must divide local_seq_len={local_seq_len}"

            total_seq_len = local_seq_len * world_size
            num_chunks = total_seq_len // chunk_size

            attn_out_local = torch.empty_like(input_x_local)
            cache = fla_Cahce.from_legacy_cache()

            for chunk_id in range(num_chunks):
                if world_size > 1:
                    owner_rank, owner_local_start, owner_local_end = _get_owner_chunk_info(
                        chunk_id=chunk_id,
                        chunk_size=chunk_size,
                        local_seq_len=local_seq_len
                    )

                    if rank == owner_rank:
                        x_chunk = input_x_local[:, owner_local_start:owner_local_end, :].contiguous()
                    else:
                        x_chunk = None

                    x_chunk = _broadcast_from_owner(
                        x_chunk=x_chunk,
                        owner_rank=owner_rank,
                        chunk_shape=(B, chunk_size, hidden_dim),
                        dtype=input_x_local.dtype,
                        device=input_x_local.device,
                        group=self.context_group,
                    )

                    out_chunk, _, cache = self.gated_delta(
                        x_chunk,
                        past_key_values=cache,
                        use_cache=True
                    )

                    o_seq_chunk = _all_gather_seq_chunk(out_chunk, self.context_group,  q_len=x_chunk.shape[1])

                    if rank == owner_rank:
                        attn_out_local[:, owner_local_start:owner_local_end, :] = o_seq_chunk

                else:
                    start = chunk_id * chunk_size
                    end = start + chunk_size
                    x_chunk = input_x_local[:, start:end, :].contiguous()

                    out_chunk, _, cache = self.gated_delta(
                        x_chunk,
                        past_key_values=cache,
                        use_cache=True
                    )

                    attn_out_local[:, start:end, :] = out_chunk

            attn_out = attn_out_local
        else:
            attn_out = self.self_attn(input_x_local, f, freqs, L=L)

        x = self.gate(x, gate_msa, attn_out)
        del gate_msa, attn_out
        # cross-attention
        attn_out = self.cross_attn(self.cross_attn_norm(x), context, attn_mask=context_mask)
        x.add_(attn_out)
        del attn_out

        shift = get_mod_chunk(4)
        scale = get_mod_chunk(3)
        gate  = get_mod_chunk(5)

        def chunked_ffn(input_x, shift, scale, gate, chunk_size):
            B, N, D = input_x.shape
            out = torch.empty_like(input_x)

            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)

                x_chunk = input_x[:, start:end, :]
                if shift.shape[1] == 1:
                    shift_chunk = shift
                else:
                    shift_chunk = shift[:, start:end, :]
                if scale.shape[1] == 1:
                    scale_chunk = scale
                else:
                    scale_chunk = scale[:, start:end, :]
                if gate.shape[1] == 1:
                    gate_chunk = gate
                else:
                    gate_chunk = gate[:, start:end, :]

                inp_chunk = modulate(self.ffn_norm(x_chunk), shift_chunk, scale_chunk)
                out[:, start:end, :] = self.gate(x_chunk, gate_chunk, self.ffn(inp_chunk))
            return out

        chunk_size = 2310
        x = chunked_ffn(x, shift, scale, gate, chunk_size)
        del shift, scale, gate

        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            ACT2CLS['silu'](),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


def build_2d_sincos_pos_embed(embed_dim: int, h: int, w: int, device=None, dtype=None):
    """
    (1, h*w, embed_dim)  2D sin-cos positional embedding.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4."
    device = device or "cpu"
    dtype = dtype or torch.float32

    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (h, w)
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    dim_each = embed_dim // 2
    omega = torch.arange(dim_each // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (dim_each // 2)))

    out_y = yy[:, None] * omega[None, :]
    out_x = xx[:, None] * omega[None, :]

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # (h*w, dim_each)
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # (h*w, dim_each)

    pos = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)  # (1, h*w, embed_dim)
    return pos



class PosEmbed2D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, h: int, w: int):
        # x: (B, N, C), N = h*w
        B, N, C = x.shape
        assert C == self.embed_dim and N == h * w
        pos = build_2d_sincos_pos_embed(C, h, w, device=x.device, dtype=torch.float32)
        return x + pos.to(x.dtype)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


@DITS.register_module()
class KairosDiT(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        dilated_lengths = [1, 1, 6, 1],
        use_first_frame_cond: bool = False,
        use_seq_parallel=False,
        use_tp_in_getaeddeltanet=False,
        use_tp_in_self_attn=False,
        attend_k0=False
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.use_first_frame_cond = use_first_frame_cond

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            ACT2CLS['silu'](),
            nn.Linear(dim, dim),
        )
        if self.use_first_frame_cond:
            # logging_once(f'use_first_frame_cond: {use_first_frame_cond}')
            self.image_downsample = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, stride=2, padding=1),
                nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            )
            self.image_embedding = MLP(dim, dim, has_pos_emb=False)
            self.image_pos_embed = PosEmbed2D(dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        _blocks = []

        use_linear_attns = [(i + 1) % 4 == 0 for i in range(num_layers)]
        gateddeltanet_layer_indexs = [-1 for _ in range(num_layers)]
        gidx = 0
        for i, vi in enumerate(use_linear_attns):
            if vi:
                gateddeltanet_layer_indexs[i] = gidx
                gidx += 1

        for i in range(num_layers):
            _block = DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, use_linear_attn=(i + 1) % 4 == 0, dilated_length=dilated_lengths[i % 4],
                    block_idx=i,
                    gateddeltanet_layer_idx=gateddeltanet_layer_indexs[i],
                    is_first_block=(i == 0),
                    is_last_block= (i == (num_layers - 1)),
                    use_seq_parallel=use_seq_parallel,
                    use_tp_in_getaeddeltanet=use_tp_in_getaeddeltanet,
                    use_tp_in_self_attn=use_tp_in_self_attn,
                    attend_k0=attend_k0
            )
            _blocks.append(_block)
        self.blocks = nn.ModuleList(_blocks)
        print(f'{self.__class__.__name__} use_seq_parallel: {use_seq_parallel}')
        print(f'{self.__class__.__name__} use_tp_in_getaeddeltanet: {use_tp_in_getaeddeltanet}')
        print(f'{self.__class__.__name__} use_tp_in_self_attn: {use_tp_in_self_attn}')
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                first_frame_latents: Optional[torch.Tensor] = None,
                **kwargs,
                ):

        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if first_frame_latents is not None and self.use_first_frame_cond:
            # shape: (b, c, t, h, w)
            first_frame_latents = first_frame_latents.to(context.device)
            img_context = self.image_downsample(first_frame_latents.squeeze(2))
            fb, fc, fh, fw = img_context.shape
            img_context = img_context.flatten(2).transpose(-2, -1)
            img_context = self.image_embedding(img_context)
            img_context = self.image_pos_embed(img_context, h=fh, w=fw)
            context = torch.cat([img_context, context], dim=1)
            if context_mask is not None:
                context_mask = torch.cat([
                    torch.ones(context.shape[0], img_context.shape[1], dtype=context_mask.dtype, device=context_mask.device),
                    context_mask
                ], dim=1)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)
        grid_size = (f, h, w)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, grid_size, context_mask=context_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, grid_size, context_mask=context_mask)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
