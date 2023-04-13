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

from mindspore import Tensor
from mindspore.nn import Cell

from mindsponge.function import get_integer
from mindsponge.data import get_hyper_string

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
                 n_hidden: int = 1,
                 ):

        super().__init__()

        self.dim_in = get_integer(dim_in)
        self.dim_out = get_integer(dim_out)
        self.activation = activation
        self.n_hidden = get_integer(n_hidden)

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
                 ):

        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
            n_hidden=n_hidden,
        )

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
                 ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            activation=activation,
            n_hidden=n_hidden,
        )

        self.linear = Dense(self.dim_in, self.dim_out, activation=None)
        self.residual = Residual(
            self.dim_out, activation=self.activation, n_hidden=n_hidden)

    def construct(self, x: Tensor):
        lx = self.linear(x)
        return self.residual(lx)



_FILTER_BY_NAME = {filter.__name__: filter for filter in _FILTER_BY_KEY.values()}


def get_filter(filter: str,
               dim_in: int,
               dim_out: int,
               activation: Cell = None,
               n_hidden: int = 1,
               ) -> Filter:
    """get filter by name"""

    if isinstance(filter, Filter):
        return filter
    if filter is None:
        return None

    hyper_param = None
    if isinstance(filter, dict):
        if 'name' not in filter.keys():
            raise KeyError('Cannot find the key "name" in filter dict!')
        hyper_param = filter
        filter = get_hyper_string(hyper_param, 'name')

    if isinstance(filter, str):
        if filter.lower() == 'none':
            return None
        if filter.lower() in _FILTER_BY_KEY.keys():
            return _FILTER_BY_KEY[filter.lower()](dim_in=dim_in,
                                               dim_out=dim_out,
                                               activation=activation,
                                               n_hidden=n_hidden)
        if filter in _FILTER_BY_NAME.keys():
            return _FILTER_BY_NAME[filter](dim_in=dim_in,
                                        dim_out=dim_out,
                                        activation=activation,
                                        n_hidden=n_hidden)

        raise ValueError(
            "The filter corresponding to '{}' was not found.".format(filter))

    raise TypeError("Unsupported filter type '{}'.".format(type(filter)))
