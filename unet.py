from typing import Callable, Literal, Union

import torch
from torch import Tensor
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A_pinv: Callable[[Tensor], Tensor], depth: int = 4, base_channels: int = 64, *, dims: Union[Literal[1],Literal[2],Literal[3]] = 2) -> None:
        super().__init__()
        self.__A_pinv = A_pinv
        ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        TransposeConvType = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dims-1]
        MaxPoolType = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dims-1]
        self.__down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvType(in_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ]+[
                nn.Sequential(
                    MaxPoolType(2),
                    ConvType(base_channels*2**(i-1), base_channels*2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels*2**i, base_channels*2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                ) for i in range(1, depth)
            ]
        )
        self.__central_block = nn.Sequential(
            MaxPoolType(2),
            ConvType(base_channels*2**(depth-1), base_channels*2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvType(base_channels*2**depth, base_channels*2**depth, kernel_size=3, padding=1),
            nn.ReLU(),
            TransposeConvType(base_channels*2**depth, base_channels*2**(depth-1), kernel_size=2, stride=2),
        )
        self.__up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvType(base_channels*2**(i+1), base_channels*2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels*2**i, base_channels*2**i, kernel_size=3, padding=1),
                    nn.ReLU(),
                    TransposeConvType(base_channels*2**i, base_channels*2**(i-1), kernel_size=2, stride=2),
                ) for i in range(depth-1,0,-1)
            ]+[
                nn.Sequential(
                    ConvType(base_channels*2, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    ConvType(base_channels, out_channels, kernel_size=1),
                )
            ]
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.__A_pinv(x)
        assert x.shape[-1]%2**len(self.__down_blocks) == 0, "Invalid input size"
        tmp = []
        delta = x
        for down_block in self.__down_blocks:
            delta = down_block(delta)
            tmp.append(delta)
        delta = self.__central_block(delta)
        for i, up_block in enumerate(self.__up_blocks):
            delta = torch.cat([delta, tmp[-(i+1)]], dim=1)
            delta = up_block(delta)
        return x+delta