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
Embedding
"""

from typing import Union, Tuple

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell


from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length
from mindsponge.function import get_integer, get_ms_array

from ..activation import get_activation

_EMBEDDING_BY_KEY = dict()


def _embedding_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _EMBEDDING_BY_KEY:
            _EMBEDDING_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _EMBEDDING_BY_KEY:
                _EMBEDDING_BY_KEY[alias] = cls
        return cls
    return alias_reg


class GraphEmbedding(nn.Cell):
    r"""Base class of graph embedding network

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off distance. Default: Length(1, 'nm')

        activation: Union[Cell, str]: Activation function. Default: None

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    """
    def __init__(self,
                 dim_node: int,
                 dim_edge: int,
                 emb_dis: bool = True,
                 emb_bond: bool = False,
                 cutoff: Union[Length, float, Tensor] = Length(1, 'nm'),
                 activation: Union[Cell, str] = None,
                 length_unit: Union[str, Units] = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.emb_dis = emb_dis
        self.emb_bond = emb_bond

        self._dim_node = get_integer(dim_node)
        self._dim_edge = get_integer(dim_edge)

        self.cutoff = get_ms_array(get_length(cutoff, self.units), ms.float32)

        self.activation = get_activation(activation)

    @property
    def dim_node(self) -> int:
        r"""dimension of node embedding vectors"""
        return self._dim_node

    @property
    def dim_edge(self) -> int:
        r"""dimension of edge embedding vectors"""
        return self._dim_edge

    def convert_length_from(self, unit: Union[str, Units]) -> float:
        """returns a scale factor that converts the length from a specified unit."""
        return self.units.convert_length_from(unit)

    def convert_length_to(self, unit: Union[str, Units]) -> float:
        """returns a scale factor that converts the length to a specified unit."""
        return self.units.convert_length_to(unit)

    def construct(self,
                  atom_type: Tensor,
                  atom_mask: Tensor,
                  neigh_dis: Tensor,
                  neigh_vec: Tensor,
                  neigh_list: Tensor,
                  neigh_mask: Tensor,
                  bond: Tensor,
                  bond_mask: Tensor,
                  **kwargs,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the properties of the molecules.

        Args:
            atom_type (Tensor): Tensor of shape (B, A). Data type is int.
                Index of atom types. Default: None
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool
                Mask for atom types
            neigh_dis (Tensor): Tensor of shape (B, A, N). Data type is float.
                Distances between central atom and its neighouring atoms.
            neigh_vec (Tensor): Tensor of shape (B, A, N, D). Data type is bool.
                Vectors from central atom to its neighouring atoms.
            neigh_list (Tensor): Tensor of shape (B, A, N). Data type is int.
                Indices of neighbouring atoms.
            neigh_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for neighbour list.
            bond_types (Tensor): Tensor of shape (B, A, N). Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for bonds

        Returns:
            node_emb (Tensor): Tensor of shape (B, A, E). Data type is float.
                Node embedding vector.
            node_mask (Tensor): Tensor of shape (B, A, E). Data type is float.
                Mask for Node embedding vector.
            edge_emb (Tensor): Tensor of shape (B, A, N, K). Data type is float.
                Edge embedding vector.
            edge_mask (Tensor): Tensor of shape (B, A, N, K). Data type is float.
                Mask for edge embedding vector.
            edge_cutoff (Tensor): Tensor of shape (B, A, N). Data type is float.
                Cutoff for edge.
            edge_self (Tensor): Tensor of shape (1, K). Data type is float.
                The edge embedding vector of the atom itself.

        Symbols:
            B:  Batch size.
            A:  Number of atoms in system.
            E:  Dimension of node embedding vector
            K:  Dimension of edge embedding vector
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
