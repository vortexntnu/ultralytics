"""Load classification weights into detection model with layer index remapping.

Classification model has C2PSA at model.9, detection model has SPPF at model.9
and C2PSA at model.10. Standard intersect_dicts misses C2PSA because the key
names differ (model.9.* vs model.10.*). This remaps cls model.9.* -> model.10.*
before loading so C2PSA weights transfer correctly.

Usage:
    from callbacks import cls_to_det_remap
    model = YOLO("yolo26s.yaml")
    cls_to_det_remap.load(model, "path/to/cls-weights.pt")
    model.train(data="coco.yaml", ...)  # no pretrained= needed
"""

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import intersect_dicts


def load(model, weights_path, remap=True):
    """Load cls weights into det model with optional C2PSA index remapping.

    Args:
        model: YOLO model instance (detection architecture).
        weights_path (str): Path to classification checkpoint.
        remap (bool): Remap cls model.9 (C2PSA) to det model.10 (C2PSA).
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model") or ckpt.get("ema")
    csd = src.float().state_dict()

    if remap:
        remapped = {}
        for k, v in csd.items():
            if k.startswith("model.9."):
                remapped[k.replace("model.9.", "model.10.")] = v
            remapped[k] = v
        csd = remapped
        LOGGER.info("Remapped cls model.9 (C2PSA) -> det model.10 (C2PSA)")

    updated = intersect_dicts(csd, model.model.state_dict())
    model.model.load_state_dict(updated, strict=False)
    LOGGER.info(f"Transferred {len(updated)}/{len(model.model.state_dict())} items from pretrained weights")
