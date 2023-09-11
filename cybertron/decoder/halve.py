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

from typing import Union

from mindspore.nn import Cell

from sponge.function import get_arguments

from .decoder import Decoder, _decoder_register
from ..layer import MLP, Dense


@_decoder_register('halve')
class HalveDecoder(Decoder):
    r"""A MLP decoder with halve number of layers.

    Args:

        dim_in (int): Input dimension.

        dim_out (int): Output dimension. Default: 1

        activation (Union[Cell, str]): Activation function. Default: None

        n_layers (int): Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 activation: Union[Cell, str] = None,
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
        return 'HalveDecoder<>'
