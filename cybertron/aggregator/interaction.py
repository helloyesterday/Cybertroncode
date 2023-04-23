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
Aggregator for node vector
"""

from typing import Union, List

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from mindsponge.function import get_arguments

from ..layer import MLP, Dense

__all__ = [
    "InteractionAggregator",
    "InteractionSummation",
    "InteractionMean",
    "LinearTransformation",
    "MultipleChannelRepresentation",
    "get_interaction_aggregator",
]

_INTERACTION_AGGREGATOR_BY_KEY = dict()


def _interaction_aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _INTERACTION_AGGREGATOR_BY_KEY:
            _INTERACTION_AGGREGATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _INTERACTION_AGGREGATOR_BY_KEY:
                _INTERACTION_AGGREGATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg


class InteractionAggregator(nn.Cell):
    r"""Network to aggregate the representation of each interaction layer.

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

        self.stack = P.Stack(-1)
        self.reduce_sum = P.ReduceSum()

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        """Aggregate the representations of each interaction layer.

        Args:
            ylist (list):       List of representation of interactions layers.
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        raise NotImplementedError


@_interaction_aggregator_register('sum')
class InteractionSummation(InteractionAggregator):
    r"""A interaction aggregator to summation all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'sum'

    def __str__(self):
        return "sum"

    def construct(self, ylist, atom_mask=None):
        xt = self.stack(ylist)
        y = self.reduce_sum(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('mean')
class InteractionMean(InteractionAggregator):
    r"""A interaction aggregator to average all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'mean'
        self.reduce_mean = P.ReduceMean()

    def __str__(self):
        return "mean"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        xt = self.stack(ylist)
        y = self.reduce_mean(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('linear')
class LinearTransformation(InteractionAggregator):
    r"""A interaction aggregator to aggregate all representations of interaction layers
        by using linear transformation

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'linear'

        self.scale = ms.Parameter(initializer(
            Normal(1.0), [self.dim]), name="scale")
        self.shift = ms.Parameter(initializer(
            Normal(1.0), [self.dim]), name="shift")

    def __str__(self):
        return "linear"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        yt = self.stack(ylist)
        ysum = self.reduce_sum(yt, -1)
        y = self.scale * ysum + self.shift
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('mcr')
class MultipleChannelRepresentation(InteractionAggregator):
    r"""A Multiple-Channel Representation (MCR) interaction aggregator to
        aggregate all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int,
                 num_agg: int,
                 n_hidden: int = 0,
                 activation: Cell = None,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'mcr'

        self.dim = dim
        self.num_agg = num_agg
        self.n_hidden = n_hidden
        self.activation = activation

        sub_dim = self.dim // self.num_agg
        last_dim = self.dim - (sub_dim * (self.num_agg - 1))
        sub_dims = [sub_dim for _ in range(self.num_agg - 1)]
        sub_dims.append(last_dim)

        if self.n_hidden > 0:
            hidden_layers = [dim] * self.n_hidden
            self.mcr = nn.CellList([
                MLP(self.dim, sub_dims[i], hidden_layers,
                    activation=self.activation)
                for i in range(self.num_agg)
            ])
        else:
            self.mcr = nn.CellList([
                Dense(self.dim, sub_dims[i], activation=self.activation)
                for i in range(self.num_agg)
            ])

        self.concat = P.Concat(-1)

    def __str__(self):
        return "MCR"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        readouts = ()
        for i in range(self.num_agg):
            readouts = readouts + (self.mcr[i](ylist[i]),)
        y = self.concat(readouts)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


_INTERACTION_AGGREGATOR_BY_NAME = {
    agg.__name__: agg for agg in _INTERACTION_AGGREGATOR_BY_KEY.values()}


def get_interaction_aggregator(cls_name: Union[InteractionAggregator, str, dict],
                               **kwargs,
                               ) -> InteractionAggregator:
    """get aggregator by name"""
    if cls_name is None or isinstance(cls_name, InteractionAggregator):
        return cls_name
    if isinstance(cls_name, dict):
        return get_interaction_aggregator(**cls_name)
    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _INTERACTION_AGGREGATOR_BY_KEY.keys():
            return _INTERACTION_AGGREGATOR_BY_KEY[cls_name.lower()](**kwargs)
        if cls_name in _INTERACTION_AGGREGATOR_BY_NAME.keys():
            return _INTERACTION_AGGREGATOR_BY_NAME[cls_name](**kwargs)
        raise ValueError(
            "The Interaction Aggregator corresponding to '{}' was not found.".format(cls_name))
    raise TypeError(
        "Unsupported Interaction Aggregator type '{}'.".format(type(cls_name)))
