from __future__ import annotations
from abc import ABC, abstractmethod
from math import floor, log2
from typing import List, Optional, Union,  cast

import torch
from torch import Tensor
import torch.nn.functional as F


class LinearOperator(ABC):
    @staticmethod
    def from_matrix(matrix: Tensor) -> LinearOperator:
        return MatrixOperator(matrix)

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...

    def to(self, target: Union[str,torch.device,torch.dtype]) -> LinearOperator:
        if isinstance(target, str):
            target = torch.device(target)
        for attr_name, attr in vars(self).items():
            if isinstance(attr, (LinearOperator,torch.Tensor)):
                setattr(self, attr_name, attr.to(target))
        return self

    @property
    @abstractmethod
    def input_size(self) -> Optional[int]:
        ...
    
    @property
    @abstractmethod
    def output_size(self) -> Optional[int]:
        ...
    
    def __add__(self, other: Union[float, LinearOperator]) -> LinearOperator:
        raise NotImplementedError()  # TODO Implement
    
    def __sub__(self, other: Union[float, LinearOperator]) -> LinearOperator:
        raise NotImplementedError()  # TODO Implement
    
    def __mul__(self, f: float) -> LinearOperator:
        return ComposedOperator(ScalingOperator(f), self)
    
    def __div__(self, f: float) -> LinearOperator:
        return self*(1.0/f)
        
    def __pow__(self, i: int) -> LinearOperator:
        if i < 0:
            raise ValueError()
        if i == 0:
            return IdentityOperator()
        max_pot = floor(log2(i))
        pots: List[LinearOperator] = [self]
        for _ in range(max_pot):
            pots.append(pots[-1]@pots[-1])
        op = None
        for j in range(max_pot):
            if i&1:
                op = pots[j] if op is None else ComposedOperator(pots[j], op)
            i >>= 1
        return cast(LinearOperator, op)
    
    def __matmul__(self, op: LinearOperator) -> LinearOperator:
        return ComposedOperator(op, self)
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        raise NotImplementedError()

    @property
    def mT(self) -> LinearOperator:
        raise NotImplementedError

    @property
    def T(self) -> LinearOperator:
        return self.mT
    
    @property
    def inv(self) -> LinearOperator:
        raise NotImplementedError
    
    @property
    def pinv(self) -> LinearOperator:
        raise NotImplementedError
    

class IdentityOperator(LinearOperator):
    def __call__(self, x: Tensor) -> Tensor:
        return x

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        if input_size is None:
            raise ValueError()
        if input_size != output_size:
            raise ValueError()
        return torch.eye(input_size)

    @property
    def mT(self) -> LinearOperator:
        return self
    
    @property
    def inv(self) -> LinearOperator:
        return self
    
    @property
    def pinv(self) -> LinearOperator:
        return self


class ScalingOperator(LinearOperator):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.__scale = scale

    def __call__(self, x: Tensor) -> Tensor:
        return x*self.__scale

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        if input_size is None:
            raise ValueError()
        if input_size != output_size:
            raise ValueError()
        return torch.eye(input_size)*self.__scale

    @property
    def mT(self) -> LinearOperator:
        return self
    
    @property
    def inv(self) -> LinearOperator:
        return ScalingOperator(1.0/self.__scale)
    
    @property
    def pinv(self) -> LinearOperator:
        return ScalingOperator(1.0/self.__scale)


class MatrixOperator(LinearOperator):
    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        if matrix.dim() != 2:
            raise ValueError()
        self.__matrix = matrix

    def __call__(self, x: Tensor) -> Tensor:
        return (self.__matrix@x.unsqueeze(-1))[...,0]

    @property
    def input_size(self) -> Optional[int]:
        return self.__matrix.shape[1]
    
    @property
    def output_size(self) -> Optional[int]:
        return self.__matrix.shape[0]
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        return self.__matrix

    @property
    def mT(self) -> LinearOperator:
        return MatrixOperator(self.__matrix.mT)
    
    @property
    def inv(self) -> LinearOperator:
        return MatrixOperator(self.__matrix.inverse())
    
    @property
    def pinv(self) -> LinearOperator:
        return MatrixOperator(self.__matrix.pinverse())
    

class ComposedOperator(LinearOperator):
    def __init__(self, op1: LinearOperator, op2: LinearOperator) -> None:
        super().__init__()
        if op2.input_size is not None and op1.output_size != op2.input_size:
            raise ValueError()
        self.__op1 = op1
        self.__op2 = op2

    def __call__(self, x: Tensor) -> Tensor:
        return self.__op2(self.__op1(x))

    @property
    def input_size(self) -> Optional[int]:
        return self.__op1.input_size
    
    @property
    def output_size(self) -> Optional[int]:
        return self.__op2.output_size
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        return self.__op2.to_matrix(input_size, self.__op1.input_size)@self.__op1.to_matrix(None, output_size)

    @property
    def mT(self) -> LinearOperator:
        return ComposedOperator(self.__op2.mT, self.__op1.mT)
    
    @property
    def inv(self) -> LinearOperator:
        return ComposedOperator(self.__op2.inv, self.__op1.inv)
    
    @property
    def pinv(self) -> LinearOperator:
        return ComposedOperator(self.__op2.pinv, self.__op1.pinv)


class ConvolutionOperator(LinearOperator):
    def __init__(self, kernel: Tensor) -> None:
        super().__init__()
        if kernel.dim() != 1:
            raise ValueError()
        if kernel.shape[0]%2 != 1:
            raise ValueError()
        self.__kernel = kernel

    def __call__(self, x: Tensor) -> Tensor:
        return F.conv1d(x.unsqueeze(-2), self.__kernel.reshape(1,1,-1), padding=self.__kernel.shape[0]//2)[...,0,:]

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        if input_size is None:
            raise ValueError()
        matrix = self.__kernel.repeat(input_size,1)
        matrix = F.pad(matrix, (0,input_size-self.__kernel.shape[0]+1))
        matrix = matrix.flatten()[:-input_size].reshape(input_size,input_size)
        matrix *= torch.triu(torch.ones(input_size,input_size))  # Zero padding instead of circular
        return matrix

    @property
    def mT(self) -> LinearOperator:
        return TransposeConvolutionOperator(self.__kernel)
    
    @property
    def inv(self) -> LinearOperator:
        return InverseConvolutionOperator(self.__kernel)
    
    @property
    def pinv(self) -> LinearOperator:
        return InverseConvolutionOperator(self.__kernel)


class TransposeConvolutionOperator(LinearOperator):
    def __init__(self, kernel: Tensor) -> None:
        super().__init__()
        if kernel.dim() != 1:
            raise ValueError()
        if kernel.shape[0]%2 != 1:
            raise ValueError()
        self.__kernel = kernel

    def __call__(self, x: Tensor) -> Tensor:
        return F.conv_transpose1d(x.unsqueeze(-2), self.__kernel.reshape(1,1,-1), padding=self.__kernel.shape[0]//2)[...,0,:]

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        return ConvolutionOperator(self.__kernel).to_matrix().mT  # TODO Direct calculation

    @property
    def mT(self) -> LinearOperator:
        return ConvolutionOperator(self.__kernel)
    
    @property
    def inv(self) -> LinearOperator:
        raise NotImplementedError()  # TODO Implement
    
    @property
    def pinv(self) -> LinearOperator:
        raise NotImplementedError()  # TODO Implement


class InverseConvolutionOperator(LinearOperator):
    def __init__(self, kernel: Tensor) -> None:
        super().__init__()
        if kernel.dim() != 1:
            raise ValueError()
        if kernel.shape[0]%2 != 1:
            raise ValueError()
        self.__kernel = kernel

    def __call__(self, x: Tensor) -> Tensor:
        x_padded = torch.nn.functional.pad(x, (1,1))
        kernel_padded = torch.nn.functional.pad(self.__kernel, (0,x.shape[1]-self.__kernel.shape[1]+2))
        x_fourier = torch.fft.fft(x_padded)
        kernel_fourier = torch.fft.fft(kernel_padded)
        kernel_fourier.imag *= -1  # Why?
        result_fourier = x_fourier/kernel_fourier
        return torch.fft.ifft(result_fourier).real[:,0:-2]

    @property
    def input_size(self) -> Optional[int]:
        return None
    
    @property
    def output_size(self) -> Optional[int]:
        return None
    
    def to_matrix(self, input_size: Optional[int] = None, output_size: Optional[int] = None) -> Tensor:
        return ConvolutionOperator(self.__kernel).to_matrix().inverse()  # TODO Direct calculation

    @property
    def mT(self) -> LinearOperator:
        raise NotImplementedError()  # TODO Implement
    
    @property
    def inv(self) -> LinearOperator:
        return ConvolutionOperator(self.__kernel)
    
    @property
    def pinv(self) -> LinearOperator:
        return ConvolutionOperator(self.__kernel)
