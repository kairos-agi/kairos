import copy
import torch.nn as nn
from mmengine.registry import Registry


PIPELINES_API = Registry('model_pipeline')
KAIROS_PROCESSOR = Registry('kairos_processor')
DITS = Registry('dits')

def build_model_pipeline(cfg):
    # Ensure registries are populated in a clean process.
    import kairos.modules.dits  # noqa: F401
    import kairos.pipelines  # noqa: F401

    model_cls = PIPELINES_API.get(cfg['type'])
    if model_cls is None:
        raise KeyError(f"unknown model pipeline type: {cfg['type']!r}")
    _cfg = copy.deepcopy(cfg)
    _cfg.pop('type')
    return model_cls(config=_cfg)

