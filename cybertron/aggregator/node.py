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
Aggregator for interaction layer
"""

from typing import Union

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer

from mindsponge.function import get_integer, get_arguments

from ..layer import Dense
from ..base import SoftmaxWithMask
from ..base import MultiheadAttention

__all__ = [
    "NodeAggregator",
    "TensorSummation",
    "TensorMean",
    "SoftmaxGeneralizedAggregator",
    "PowermeanGeneralizedAggregator",
    "get_node_aggregator",
]

_AGGREGATOR_BY_KEY = dict()


def _aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _AGGREGATOR_BY_KEY:
            _AGGREGATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _AGGREGATOR_BY_KEY:
                _AGGREGATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg


class NodeAggregator(nn.Cell):
    r"""Network to aggregate the outputs of each atoms.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = kwargs

        self.dim = get_integer(dim)
        self.axis = get_integer(axis)

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """Aggregate the outputs of each atoms.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        raise NotImplementedError

    def __str__(self):
        return 'Aggregator<>'


@_aggregator_register('sum')
class TensorSummation(NodeAggregator):
    r"""A aggregator to sum all the tensors.

    Args:

        dim (int):  Feature dimension. Default: None

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int = None,
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            axis=axis,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To sum all tensors.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if atom_mask is not None:
            # (B,A,X) * (B,A,1)
            nodes = nodes * F.expand_dims(atom_mask, -1)
        # (B,X) <- (B,A,X)
        agg = F.reduce_sum(nodes, self.axis)
        return agg

    def __str__(self):
        return "TensorSummation<>"


@_aggregator_register('mean')
class TensorMean(NodeAggregator):
    r"""A aggregator to average all the tensors.

    Args:

        dim (int):  Feature dimension. Default: None

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int = None,
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            axis=axis,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To average all tensors.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if atom_mask is None:
            # (B,X) <- (B,A,X)
            return F.reduce_mean(nodes, self.axis)

        # (B,A,X) * (B,A,1)
        nodes = nodes * F.expand_dims(atom_mask, -1)
        # (B,X) <- (B,A,X)
        agg = F.reduce_sum(nodes, self.axis)
        # (B,X) / (B,1)
        return agg / num_atoms

    def __str__(self):
        return "TensorMean<>"


@_aggregator_register('softmax')
class SoftmaxGeneralizedAggregator(NodeAggregator):
    r"""A Softmax-based generalized mean-max-sum aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            axis=axis,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('zero', 1), name="rho")

        self.softmax = ops.Softmax(axis=self.axis)
        self.softmax_with_mask = SoftmaxWithMask(axis=self.axis)

        self.expand_ones = F.ones((1, 1, self.dim), ms.int32)

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using softmax-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if num_atoms is None:
            num_atoms = nodes.shape[self.axis]

        scale = num_atoms / (1 + self.beta * (num_atoms - 1))
        px = nodes * self.rho

        if atom_mask is None:
            agg_nodes = self.softmax(px) * nodes
        else:
            atom_mask = F.expand_dims(atom_mask, -1)
            mask = (self.expand_ones * atom_mask) > 0
            agg_nodes = self.softmax_with_mask(px, mask) * nodes * atom_mask

        agg_nodes = F.reduce_sum(agg_nodes, self.axis)

        return scale * agg_nodes

    def __str__(self):
        return "SoftmaxGeneralizedAggregator<>"


@_aggregator_register('powermean')
class PowermeanGeneralizedAggregator(NodeAggregator):
    r"""A PowerMean-based generalized mean-max-sum aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            axis=axis,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('one', 1), name="rho")

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using PowerMean-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if num_atoms is None:
            num_atoms = nodes.shape[self.axis]

        scale = num_atoms / (1 + self.beta * (num_atoms - 1))
        xp = F.pow(nodes, self.rho)
        if atom_mask is not None:
            xp = xp * F.expand_dims(atom_mask, -1)
        agg_nodes = F.reduce_sum(xp, self.axis)

        return F.pow(scale*agg_nodes, 1.0/self.rho)

    def __str__(self):
        return "PowermeanGeneralizedAggregator<>"


@_aggregator_register('transformer')
class TransformerAggregator(NodeAggregator):
    r"""A transformer-type aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2,
                 n_heads: int = 8,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            axis=axis,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.a2q = Dense(dim, dim, has_bias=False)
        self.a2k = Dense(dim, dim, has_bias=False)
        self.a2v = Dense(dim, dim, has_bias=False)

        self.layer_norm = nn.LayerNorm((dim,), -1, -1)

        self.multi_head_attention = MultiheadAttention(
            dim, n_heads, dim_tensor=3)

        self.mean = TensorMean(dim, axis)

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using PowerMean-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """

        # [B, A, F]
        x = self.layer_norm(nodes)

        # [B, A, F]
        query = self.a2q(x)
        key = self.a2k(x)
        value = self.a2v(x)

        # [B, A, F]
        x = self.multi_head_attention(query, key, value, atom_mask)

        # [B, 1, F]
        return self.mean(x, atom_mask, num_atoms)
    
    def __str__(self):
        return "TransformerAggregator<>"


_AGGREGATOR_BY_NAME = {
    agg.__name__: agg for agg in _AGGREGATOR_BY_KEY.values()}


def get_node_aggregator(cls_name: Union[NodeAggregator, str, dict],
                        dim: int,
                        axis: int = -2,
                        **kwargs,
                        ) -> NodeAggregator:
    """get aggregator by name"""
    if cls_name is None or isinstance(cls_name, NodeAggregator):
        return cls_name
    if isinstance(cls_name, dict):
        return get_node_aggregator(**cls_name)
    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _AGGREGATOR_BY_KEY.keys():
            return _AGGREGATOR_BY_KEY[cls_name.lower()](dim=dim, axis=axis, **kwargs)
        if cls_name in _AGGREGATOR_BY_NAME.keys():
            return _AGGREGATOR_BY_NAME[cls_name](dim=dim, axis=axis, **kwargs)
        raise ValueError(
            "The Aggregator corresponding to '{}' was not found.".format(cls_name))
    raise TypeError(
        "Unsupported Aggregator type '{}'.".format(type(cls_name)))
