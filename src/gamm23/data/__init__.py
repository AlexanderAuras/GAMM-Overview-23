from pathlib import Path

from typing import Tuple, Union, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset



def load_datasets(base_path: Union[str,Path]) -> Tuple[Dataset[Tuple[Tensor,Tensor]],Dataset[Tuple[Tensor,Tensor]],Dataset[Tuple[Tensor,Tensor]]]:
    base_path = Path(base_path)
    train_dataset = TensorDataset(*torch.load(str(base_path.joinpath("train.pth").resolve())))
    val_dataset = TensorDataset(*torch.load(str(base_path.joinpath("val.pth").resolve())))
    test_dataset = TensorDataset(*torch.load(str(base_path.joinpath("test.pth").resolve())))
    return cast(Tuple[Dataset[Tuple[Tensor,Tensor]],Dataset[Tuple[Tensor,Tensor]],Dataset[Tuple[Tensor,Tensor]]], (train_dataset, val_dataset, test_dataset))

