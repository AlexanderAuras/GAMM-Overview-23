from typing import Optional

import torch
from torch import Tensor

from gamm23.operators.linear_op import LinearOperator, ConvolutionOperator


class IntegrationOperator(LinearOperator):
    def __call__(self, x: Tensor) -> Tensor:
        return x.cumsum(-1)

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        if input_size is None or output_size is None:
            raise ValueError()
        if input_size != output_size:
            raise ValueError()
        return torch.tril(torch.ones((input_size,output_size)))

    @property
    def mT(self) -> LinearOperator:
        return TransposeIntegrationOperator()
    
    @property
    def inv(self) -> LinearOperator:
        return DifferentialOperator()
    
    @property
    def pinv(self) -> LinearOperator:
        return DifferentialOperator()


class TransposeIntegrationOperator(LinearOperator):
    def __call__(self, x: Tensor) -> Tensor:
        return x.flip(-1).cumsum(-1).flip(-1)

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        if input_size is None or output_size is None:
            raise ValueError()
        if input_size != output_size:
            raise ValueError()
        return torch.triu(torch.ones((input_size,output_size)))

    @property
    def mT(self) -> LinearOperator:
        return IntegrationOperator()
    
    @property
    def inv(self) -> LinearOperator:
        return DifferentialOperator().mT
    
    @property
    def pinv(self) -> LinearOperator:
        return DifferentialOperator().mT


class DifferentialOperator(ConvolutionOperator):
    def __init__(self) -> None:
        super().__init__(torch.tensor([-1.0,1.0,0.0]))