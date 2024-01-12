from typing import Callable, Literal, Union

from torch import Tensor

from gamm23.networks.unet import UNet



class ItNet(UNet):
    def __init__(self, in_channels: int, out_channels: int, A: Callable[[Tensor],Tensor], A_T: Callable[[Tensor],Tensor], A_pinv: Callable[[Tensor],Tensor], iterations: int, lr: float, depth: int = 4, base_channels: int = 64, *, dims: Union[Literal[1],Literal[2],Literal[3]] = 2) -> None:
        super().__init__(in_channels, out_channels, lambda x: x, depth, base_channels, dims=dims)
        self.__A = A
        self.__A_T = A_T
        self.__A_pinv = A_pinv
        self.__iterations = iterations
        self.__lr = lr
    
    def forward(self, y: Tensor) -> Tensor:  # type: ignore
        x = self.__A_pinv(y)
        for _ in range(self.__iterations):
            x = super().forward(x)
            x = x - self.__lr*self.__A_T(self.__A(x)-y)
        return x