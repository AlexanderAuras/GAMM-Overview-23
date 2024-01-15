from pathlib import Path

import torch


torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)


BASE_PATH = Path(__file__).parent
DATA_SIZE = 1024
MEASUREMENT_SIZE = 512

matrix = 0.05*torch.randn((MEASUREMENT_SIZE,DATA_SIZE))
torch.save(matrix, str(BASE_PATH.joinpath("matrix.pth").resolve()))