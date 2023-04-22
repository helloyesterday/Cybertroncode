# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Decoder networks for readout function
"""

from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell

from mindsponge.function import get_ms_array, get_arguments

from .block import MLP, Dense
from .block import PreActResidual
from .block import SeqPreActResidual
from .block import PreActDense
from .activation import get_activation

__all__ = [
    "Decoder",
    "get_decoder",
    "HalveDecoder",
    "ResidualOutputBlock",
]

_DECODER_BY_KEY = dict()


def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_BY_KEY:
            _DECODER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _DECODER_BY_KEY:
                _DECODER_BY_KEY[alias] = cls
        return cls
    return alias_reg


class Decoder(Cell):
    r"""Decoder network to reduce the dimension of representation

    Args:

        dim_in (int):         Input dimension.

        dim_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = kwargs

        self.reg_key = 'none'
        self.name = 'decoder'

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layers = n_layers

        self.output = None
        self.activation = get_activation(activation)

    def construct(self, x: Tensor):
        #pylint: disable=not-callable
        return self.output(x)


@_decoder_register('halve')
class HalveDecoder(Decoder):
    r"""A MLP decoder with halve number of layers.

    Args:

        dim_in (int):         Input dimension.

        dim_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 **kwargs,
                 ):

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
            n_layers=n_layers,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'halve'
        self.name = 'halve'

        if self.n_layers > 0:
            n_hiddens = []
            dim = self.dim_in
            for _ in range(self.n_layers):
                dim = dim // 2
                if dim < dim_out:
                    raise ValueError(
                        "The dimension of hidden layer is smaller than output dimension")
                n_hiddens.append(dim)
            self.output = MLP(self.dim_in, self.dim_out, n_hiddens, activation=self.activation)
        else:
            self.output = Dense(self.dim_in, self.dim_out, activation=self.activation)

    def __str__(self):
        return 'halve'


@_decoder_register('residual')
class ResidualOutputBlock(Decoder):
    r"""Residual block type decoder

    Args:

        dim_in (int):         Input dimension.

        dim_out (int):        Output dimension. Default: 1

        activation (Cell):  Activation function. Default: None

        n_layers (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 activation: Cell = None,
                 n_layers: int = 1,
                 **kwargs,
                 ):

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
            n_layers=n_layers,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'residual'
        self.name = 'residual'

        if self.n_layers == 1:
            output_residual = PreActResidual(self.dim_in, activation=self.activation)
        else:
            output_residual = SeqPreActResidual(
                self.dim_in, activation=self.activation, n_res=self.n_layers)

        self.output = nn.SequentialCell([
            output_residual,
            PreActDense(self.dim_in, self.dim_out, activation=self.activation),
        ])

    def __str__(self):
        return 'residual'


_DECODER_BY_NAME = {
    decoder.__name__: decoder for decoder in _DECODER_BY_KEY.values()}


def get_decoder(cls_name: str,
                dim_in: int,
                dim_out: int,
                activation: Cell = None,
                n_layers: int = 1,
                **kwargs
                ) -> Decoder:
    """get decoder by name"""
    if cls_name is None or isinstance(cls_name, Decoder):
        return cls_name
    
    if isinstance(cls_name, dict):
        return get_decoder(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _DECODER_BY_KEY.keys():
            return _DECODER_BY_KEY[cls_name.lower()](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                **kwargs
            )
        if cls_name in _DECODER_BY_NAME.keys():
            return _DECODER_BY_NAME[cls_name](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                **kwargs
            )

        raise ValueError(
            "The Decoder corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported init type '{}'.".format(type(cls_name)))
