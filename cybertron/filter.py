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
Filter networks
"""

from typing import Union

from mindspore import Tensor
from mindspore.nn import Cell

from mindsponge.function import get_integer, get_arguments

from .block import MLP, Dense, Residual

_FILTER_BY_KEY = dict()


def _filter_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _FILTER_BY_KEY:
            _FILTER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _FILTER_BY_KEY:
                _FILTER_BY_KEY[alias] = cls
        return cls
    return alias_reg


class Filter(Cell):
    r"""Base class for filter network.

    Args:

        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: None

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: Cell = None,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.dim_in = get_integer(dim_in)
        self.dim_out = get_integer(dim_out)
        self.activation = activation

    def construct(self, x: Tensor):
        return x


@_filter_register('dense')
class DenseFilter(Filter):
    r"""Dense type filter network.

    Args:

        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: None

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 **kwargs,
                 ):

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if n_hidden > 0:
            hidden_layers = [self.dim_out for _ in range(n_hidden)]
            self.dense_layers = MLP(
                self.dim_in, self.dim_out, hidden_layers, activation=self.activation)
        else:
            self.dense_layers = Dense(
                self.dim_in, self.dim_out, activation=self.activation)

    def construct(self, x: Tensor):
        return self.dense_layers(x)


@_filter_register('residual')
class ResFilter(Filter):
    r"""Residual type filter network.

    Args:

        dim_in (int):    Number of basis functions.

        dim_out (int):   Dimension of output filter Tensor.

        activation (Cell):  Activation function. Default: None

        n_hidden (int):     Number of hidden layers. Default: 1

    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 activation: Cell,
                 n_hidden: int = 1,
                 **kwargs,
                 ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.linear = Dense(self.dim_in, self.dim_out, activation=None)
        self.residual = Residual(
            self.dim_out, activation=self.activation, n_hidden=n_hidden)

    def construct(self, x: Tensor):
        lx = self.linear(x)
        return self.residual(lx)


_FILTER_BY_NAME = {filter.__name__: filter for filter in _FILTER_BY_KEY.values()}


def get_filter(cls_name: Union[Filter, str],
               dim_in: int,
               dim_out: int,
               activation: Cell = None,
               **kwargs,
               ) -> Filter:
    """get filter by name"""

    if isinstance(cls_name, Filter):
        return cls_name

    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_filter(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _FILTER_BY_KEY.keys():
            return _FILTER_BY_KEY[cls_name.lower()](dim_in=dim_in,
                                               dim_out=dim_out,
                                               activation=activation,
                                               **kwargs)
        if cls_name in _FILTER_BY_NAME.keys():
            return _FILTER_BY_NAME[cls_name](dim_in=dim_in,
                                        dim_out=dim_out,
                                        activation=activation,
                                        **kwargs)

        raise ValueError(
            "The filter corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported filter type '{}'.".format(type(cls_name)))
