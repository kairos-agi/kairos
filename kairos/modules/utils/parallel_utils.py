import torch
import torch.distributed as dist

class SimpleParallelState:
    """
    A `parallel_state` shim used by the inference code.

    This helper defines how different distributed process groups are used in this repo:

    1) WORLD group (global process group)
       - `dist.group.WORLD`
       - `world_size = dist.get_world_size()`

    2) VAE group (NO subgrouping; always WORLD)
       - Convention: VAE-related collectives run on the full WORLD group.
       - We do NOT split VAE into sub-groups.
       - Therefore `_vae_group` is expected to be `dist.group.WORLD` in practice.

    3) TP group (also used as the "context parallel" group here)
       - Convention: this implementation uses "context parallel" as
         "the group on which the model is sharded / coordinated".
       - If TP is enabled, `init_tp_groups(tp_size)` partitions WORLD into multiple
         contiguous TP groups, each of size `tp_size`.
       - If TP is disabled (tp_size == 1 or `_tp_group` is not set), the context
         group falls back to WORLD.
       - In other words: in this repo, `tp_group` and `cp_group` refer to the same
         group, and they have the same size (= tp_size).

    4) CFG group (ONLY needed by DiT when CFG-parallel is enabled)
       - Important: CFG sub-groups are NOT built/managed by this class.
         They are typically constructed externally via `dist.new_group(...)` and
         stored elsewhere (e.g., `parallel_state.cfg_group`).
       - When CFG-parallel is ON, WORLD is viewed as a 2D grid:
           WORLD size = tp_size * cfg_size  (cfg_size is typically 2)
         * TP group size  = tp_size  (this class' context/TP group)
         * CFG group size = cfg_size (pairs cond/uncond ranks for synchronization)
       - When CFG-parallel is OFF, no CFG sub-group is required.

    Summary:
    - VAE: always WORLD, no subgrouping.
    - DiT: CFG OFF -> only TP/context group is needed;
           CFG ON  -> an additional CFG sub-group is required.
    - This class does not implement true CP subgrouping; it simply reuses TP as
      the "context parallel" group and falls back to WORLD when TP is absent.
    """
    _tp_group = None
    _vae_group = None

    _cfg_group = None
    _cfg_rank = 0
    _cfg_size = 1

    @staticmethod
    def init_tp_groups(tp_size=4):
        assert dist.is_initialized()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert world % tp_size == 0
        num_groups = world // tp_size

        tp_groups = []
        for gid in range(num_groups):
            ranks = list(range(gid * tp_size, (gid + 1) * tp_size))
            g = dist.new_group(ranks=ranks)
            tp_groups.append((ranks, g))

        tp_group_id = rank // tp_size
        tp_rank = rank % tp_size
        tp_ranks, tp_group = tp_groups[tp_group_id]
        return tp_group, tp_rank, tp_group_id

    @classmethod
    def set_tp_group(cls, g):
        cls._tp_group = g

    @classmethod
    def set_vae_group(cls, g):
        cls._vae_group = g

    @classmethod
    def get_vae_parallel_group(cls):
        return cls._vae_group

    @classmethod
    def reset_cfg(cls):
        cls._cfg_group = None
        cls._cfg_rank = 0
        cls._cfg_size = 1

    @classmethod
    def set_cfg_group(cls, g, cfg_rank=0, cfg_size=1):
        cls._cfg_group = g
        cls._cfg_rank = int(cfg_rank)
        cls._cfg_size = int(cfg_size)

    @classmethod
    def get_cfg_parallel_group(cls):
        return cls._cfg_group

    @classmethod
    def get_cfg_parallel_rank(cls):
        return cls._cfg_rank

    @classmethod
    def get_cfg_parallel_world_size(cls):
        if (not dist.is_initialized()) or (cls._cfg_group is None):
            return 1
        return dist.get_world_size(cls._cfg_group)

    @classmethod
    def get_cfg_size(cls):
        return cls._cfg_size

    @classmethod
    def get_context_parallel_group(cls):
        if dist.is_initialized():
            return cls._tp_group if cls._tp_group is not None else dist.group.WORLD
        return None

    @classmethod
    def get_context_parallel_world_size(cls):
        if dist.is_initialized():
            g = cls.get_context_parallel_group()
            return dist.get_world_size(g)
        return 1

    @classmethod
    def get_context_parallel_rank(cls):
        g = cls.get_context_parallel_group()
        if dist.is_initialized():
            return dist.get_rank(g)
        return 0

parallel_state = SimpleParallelState()