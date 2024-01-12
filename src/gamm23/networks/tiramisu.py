from typing import Literal, Sequence, Union

import torch
from torch import Tensor
import torch.nn as nn



class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, layer_count: int, growth_rate: int, dims: int = 2) -> None:
        super().__init__()
        ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        BatchNormType = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dims-1]
        DropoutType = [nn.Dropout1d, nn.Dropout2d, nn.Dropout3d][dims-1]
        self.__layers = nn.ModuleList([
            nn.Sequential(
                BatchNormType(in_channels+i*growth_rate),
                nn.ReLU(),
                ConvType(in_channels+i*growth_rate, growth_rate, kernel_size=3, padding=1),
                DropoutType(p=0.2),
            ) for i in range(layer_count)
        ])
        
    def forward(self, x: torch.Tensor) -> Tensor:
        out = []
        tmp = [x]
        for layer in self.__layers:
            z = layer(torch.cat(tmp, dim=1))
            tmp.append(z)
            out.append(z)
        return torch.cat(out, dim=1)



class _TransitionDown(nn.Module):
    def __init__(self, in_channels: int, dims: int = 2) -> None:
        super().__init__()
        ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        MaxPoolType = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d][dims-1]
        BatchNormType = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dims-1]
        DropoutType = [nn.Dropout1d, nn.Dropout2d, nn.Dropout3d][dims-1]
        self.__layers = nn.Sequential(
            BatchNormType(in_channels),
            nn.ReLU(),
            ConvType(in_channels, in_channels, kernel_size=1),
            DropoutType(0.2),
            MaxPoolType(2),
        )
    
    def forward(self, x: torch.Tensor) -> None:
        return self.__layers(x)
        


class _TransitionUp(nn.Module):
    def __init__(self, in_channels: int, dims: int = 2) -> None:
        super().__init__()
        TransposeConvType = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][dims-1]
        self.__layers = TransposeConvType(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x: torch.Tensor) -> None:
        return self.__layers(x)



class Tiramisu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, M: int, N: int, layer_counts: Sequence[int] = (4,5,7,10,12,15), growth_rate: int = 16, base_channels: int = 48, *, dims: Union[Literal[1],Literal[2],Literal[3]] = 2) -> None:
        super().__init__()
        ConvType = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        channels_down = [48+sum(layer_counts[:i])*16 for i in range(len(layer_counts))]
        channels_up1 = [layer_counts[::-1][i]*16 for i in range(len(layer_counts)-1)]
        channels_up2 = [channels_down[::-1][i]+layer_counts[::-1][i]*16 for i in range(len(layer_counts)-1)]
        self.__linear_map = nn.Linear(M, N, bias=False)
        self.__N = N
        self.__initial_conv = ConvType(in_channels, base_channels, kernel_size=3, padding=1)
        self.__down_dense_blocks = nn.ModuleList([_DenseBlock(channels_down[i], layer_counts[i], growth_rate, dims) for i in range(len(layer_counts)-1)])
        self.__down_transitions = nn.ModuleList([_TransitionDown(channels_down[i+1], dims) for i in range(len(layer_counts)-1)])
        self.__center_dense_block = _DenseBlock(channels_down[-1], layer_counts[-1], growth_rate, dims)
        self.__up_transitions = nn.ModuleList([_TransitionUp(channels_up1[i], dims) for i in range(len(layer_counts)-1)])
        self.__up_dense_blocks = nn.ModuleList([_DenseBlock(channels_up2[i], layer_counts[-2::-1][i], growth_rate, dims) for i in range(len(layer_counts)-1)])
        self.__final_conv = ConvType(layer_counts[0]*16, out_channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.__linear_map(x.flatten(start_dim=1)).reshape(-1,1,self.__N)
        x = self.__initial_conv(x)
        tmp = []
        for i in range(len(self.__down_dense_blocks)):
            old_x  = x
            x = self.__down_dense_blocks[i](x)
            x = torch.cat([old_x, x], dim=1)
            tmp.append(x)
            x = self.__down_transitions[i](x)
        x = self.__center_dense_block(x)
        for i in range(len(self.__up_dense_blocks)):
            x = self.__up_transitions[i](x) # Concatenation and up transition switched in original code?!
            x = torch.cat([x, tmp[::-1][i]], dim=1)
            x = self.__up_dense_blocks[i](x)
        return self.__final_conv(x)