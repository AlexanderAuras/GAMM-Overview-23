from pathlib import Path
import random
from typing import Callable, Tuple, cast

import torch
from torch import Tensor
from tqdm import trange

from gamm23.operators import MatrixOperator

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)



def generate_signals(
    N_samples: int,
    A: Callable[[Tensor],Tensor],
    noise_level: float = 0.03,
    N_signal: int = 1000,
    min_jumps: int = 10, 
    max_jumps: int = 20, 
    min_distance: int = 40, 
    boundary_width: int = 20, 
    height_std: float = 1.0,
    min_height: float = 0.2
) -> Tuple[Tensor,Tensor]:
    #Calculate positions by generating min_distance-spaced positions
    #and randomly rightshifting by the cumulatice sum
    derivative = torch.zeros((N_samples,N_signal))
    for i in trange(N_samples):
        jump_count = random.randint(min_jumps-2, max_jumps-2)
        delta = torch.rand((jump_count-1)) # One more than needed (last will be discarded, as it would lead to using the right boundary as last value, due to the normalization)
        delta = torch.cumsum((N_signal-2*boundary_width-(jump_count-2)*min_distance)*delta/delta.sum(), 0)[:-1].ceil().to(torch.long)
        jump_positions = boundary_width+min_distance*torch.arange(jump_count-2)+delta.sort()[0]
        values = height_std*torch.randn((jump_count,))
        values[torch.abs(values)<min_height] = values[torch.abs(values)<min_height].sign()*min_height
        derivative[i,torch.cat([torch.tensor([0]),jump_positions,torch.tensor([0])])] = values
    groundtruths = derivative.cumsum(1)
    measurements = A(groundtruths)
    noisy_measurement = measurements+noise_level*torch.randn_like(measurements)
    return noisy_measurement, groundtruths



matrix = torch.load(Path(__file__).parents[1].joinpath("operators", "matrix.pth"))
A = MatrixOperator(matrix)
BASE_PATH = Path(__file__).parent
N_TRAIN = 8192
N_VAL = 2048
N_TEST = 2048
NOISE_LEVEL = 0.03

train_data_clean = generate_signals(N_TRAIN, A, 0.0, cast(int, A.input_size))
val_data_clean = generate_signals(N_VAL, A, 0.0, cast(int, A.input_size))
test_data_clean = generate_signals(N_TEST, A, 0.0, cast(int, A.input_size))
train_data_noisy = generate_signals(N_TRAIN, A, NOISE_LEVEL, cast(int, A.input_size))
val_data_noisy = generate_signals(N_VAL, A, NOISE_LEVEL, cast(int, A.input_size))
test_data_noisy = generate_signals(N_TEST, A, NOISE_LEVEL, cast(int, A.input_size))

BASE_PATH.joinpath("clean").mkdir(parents=True, exist_ok=True)
torch.save(train_data_noisy, str(BASE_PATH.joinpath("clean", "train.pth").resolve()))
torch.save(val_data_noisy, str(BASE_PATH.joinpath("clean", "val.pth").resolve()))
torch.save(test_data_noisy, str(BASE_PATH.joinpath("clean", "test.pth").resolve()))

BASE_PATH.joinpath("noisy").mkdir(parents=True, exist_ok=True)
torch.save(train_data_clean, str(BASE_PATH.joinpath("noisy", "train.pth").resolve()))
torch.save(val_data_clean, str(BASE_PATH.joinpath("noisy", "val.pth").resolve()))
torch.save(test_data_clean, str(BASE_PATH.joinpath("noisy", "test.pth").resolve()))
