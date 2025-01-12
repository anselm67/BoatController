import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def img_name(tagname: str) -> str:
    m = re.search('^(.*)/tags\\.([^/]+)\\.png$', tagname)
    if m:
        return os.path.join(m.group(1), f"{m.group(2)}.jpg")
    else:
        raise ValueError(f"No image file for tag file {tagname}")


def evaluate(model: nn.Module, imgs: torch.Tensor, tags: torch.Tensor) -> float:
    outputs = model(imgs)
    outputs = (outputs > 0.5).to(torch.uint8)
    return (torch.sum(outputs == tags) / np.prod(outputs.shape)).item()
