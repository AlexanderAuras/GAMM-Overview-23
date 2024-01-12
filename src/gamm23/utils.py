from math import log10
import random
from typing import Sequence, Union

import numpy as np
import torch
from torch import Tensor


def seed(seed: int) -> None:
    if seed < 0:
        return
    seed = seed%2^32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def psnr(input_: Tensor, target: Tensor, max_diff: float = 1.0, psnr_dims: Union[Sequence[int],int] = (-2,-1), reduction_dims: Union[Sequence[int],int] = (0,)) -> Tensor:
    if isinstance(psnr_dims, int):
        psnr_dims = (psnr_dims,)
    psnr_dims = [x if x >= 0 else input_.dim()+x for x in psnr_dims]
    if isinstance(reduction_dims, int):
        reduction_dims = (reduction_dims,)
    reduction_dims = [x if x >= 0 else input_.dim()+x for x in reduction_dims]
    input_ = input_.permute(*reduction_dims, *[i for i in range(input_.dim()) if i not in psnr_dims and i not in reduction_dims], *psnr_dims)
    input_ = input_.flatten(end_dim=len(reduction_dims)-1).flatten(start_dim=-len(psnr_dims))
    target = target.permute(*reduction_dims, *[i for i in range(target.dim()) if i not in psnr_dims and i not in reduction_dims], *psnr_dims)
    target = target.flatten(end_dim=len(reduction_dims)-1).flatten(start_dim=-len(psnr_dims))
    mse = torch.pow(input_-target, 2.0).sum(dim=-1)
    multiple_psnr = 20*log10(max_diff)-10.0*torch.log10(mse)
    return multiple_psnr.mean(0)
    