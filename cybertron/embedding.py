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

from typing import Union

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import Initializer, Normal

from mindsponge.function import Units, GLOBAL_UNITS, Length
from mindsponge.function import get_integer, concat_last_dim

from .cutoff import Cutoff, get_cutoff
from .rbf import get_rbf

__all__ = [
]


class AtomEmbedding(nn.Cell):
    r"""Atom embedding


    """

    def __init__(self,
                 dim_feature: int = 64,
                 num_atom_types: int = 64,
                 use_one_hot: bool = True,
                 embedding_table: Union[str, Initializer] = Normal(1.0)
                 ):

        super().__init__()

        num_atom_types = get_integer(num_atom_types)
        dim_feature = get_integer(dim_feature)
        self.embedding = nn.Embedding(
            num_atom_types, dim_feature,
            use_one_hot=use_one_hot, embedding_table=embedding_table)
        
    def construct(self, atom_types: Tensor, atom_mask: Tensor = None, batch_size: int = 1):
        atom_embedding = self.embedding(atom_types)
        if batch_size > 1 and atom_types.shape[0] != batch_size:
            atom_embedding = msnp.broadcast_to(atom_embedding, (batch_size,) + atom_embedding.shape[1:])
            if atom_mask is not None:
                atom_mask = msnp.broadcast_to(atom_mask, (batch_size,)+atom_mask.shape[1:])

        return atom_embedding, atom_mask


class DistanceEmbedding(nn.Cell):
    r"""Distance embedding

    """

    def __init__(self,
                 
                 
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf_fn: Cell = None,
                 num_basis: int = None,
                 filter: Cell = None,
                 dim_feature: int = 64,
                 r_self: Length = None,
                 length_unit: str = 'nm',
                 ):
        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.cutoff = self.get_length(cutoff)
        self.cutoff_fn = get_cutoff(cutoff_fn, self.cutoff)

        self.num_basis = get_integer(num_basis)

        self.rbf_fn = get_rbf(rbf_fn, r_max=self.cutoff, num_basis=self.num_basis,
                              length_unit=self.units.length_unit)

        self.r_self = r_self
        self.r_self_ex = None
        if self.r_self is not None:
            self.r_self = self.get_length(self.r_self)
            self.r_self_ex = F.expand_dims(self.r_self, 0)

    def set_self_distance(self, distance: Tensor):
        self.r_self = self.get_length(distance)
        self.r_self_ex = F.expand_dims(distance, 0)
        return self

    def get_length(self, length: Union[float, Length, Tensor], unit: str = None):
        """get length value according to unit"""
        if isinstance(length, Length):
            if unit is None:
                unit = self.units
            return Tensor(length(unit), ms.float32)
        return Tensor(length, ms.float32)
    
    def get_rbf(self, distances: Tensor):
        """get radical basis function"""
        if self.rbf_fn is None:
            rbf = F.expand_dims(distances, -1)
        else:
            rbf = self.rbf_fn(distances)
        return rbf

    def construct(self,
                  distances: Tensor = 1,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  atom_mask: Tensor = None,
                  ):

        rbf = self.get_rbf(distances)
        rbf_self = 0
        if self.r_self is not None:
            self.get_rbf(self.r_self_ex)

        # apply cutoff
        if self.cutoff_fn is None:
            cutoff = F.ones_like(neighbours)
        else:
            cutoff, neighbour_mask = self.cutoff_fn(neighbours, neighbour_mask)

        cutoff_self = 1
        if self.r_self is not None:
            cutoff_self = F.cast(atom_mask, ms.float32)

        return rbf, cutoff, neighbour_mask, rbf_self, cutoff_self


class BondEmbedding(nn.Cell):
    r"""Distance embedding

    """

    def __init__(self,
                 dim_feature: int = 128,
                 num_bond_types: int = 16,
                 use_one_hot: bool = True,
                 embedding_table: Union[str, Initializer] = Normal(1.0)
                 ):
        super().__init__()

        self.num_bond_types = get_integer(num_bond_types)
        self.dim_feature = get_integer(dim_feature)

        self.embedding = nn.Embedding(
            self.num_bond_types, self.dim_feature,
            use_one_hot=use_one_hot, embedding_table=embedding_table)

    def construct(self,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  atom_mask: Tensor = None,
                  ):

        nbatch = bonds.shape[0]
        natoms = bonds.shape[1]

        bond_self = F.zeros((nbatch, natoms), ms.int32)
        emb_self = self.embedding(bond_self)

        emb = self.embedding(bonds)
        cutoff = 1
        if bond_mask is not None:
            b_ij = b_ij * F.expand_dims(bond_mask, -1)
            bond_mask = concat_last_dim((atom_mask, bond_mask))
            cutoff = F.cast(bond_mask > 0, ms.float32)

        return emb, cutoff, bond_mask, emb_self



class EdgeEmbedding(nn.Cell):
    r"""Edge embedding

    """

    def __init__(self,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf_fn: Cell = None,
                 r_self: Length = None,
                 length_unit: str = 'nm',
                 ):
        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.cutoff = self.get_length(cutoff)
        self.cutoff_fn = get_cutoff(cutoff_fn, self.cutoff)
        self.rbf_fn = get_rbf(rbf_fn, self.cutoff, length_unit=self.units.length_unit)

        self.r_self = r_self
        self.r_self_ex = None
        if self.r_self is not None:
            self.r_self = self.get_length(self.r_self)
            self.r_self_ex = F.expand_dims(self.r_self, 0)

    def set_self_distance(self, distance: Tensor):
        self.r_self = self.get_length(distance)
        self.r_self_ex = F.expand_dims(distance, 0)
        return self

    def get_length(self, length: Union[float, Length, Tensor], unit: str = None):
        """get length value according to unit"""
        if isinstance(length, Length):
            if unit is None:
                unit = self.units
            return Tensor(length(unit), ms.float32)
        return Tensor(length, ms.float32)
    
    def get_rbf(self, distances: Tensor):
        """get radical basis function"""
        if self.rbf_fn is None:
            rbf = F.expand_dims(distances, -1)
        else:
            rbf = self.rbf_fn(distances)
        return rbf

    def construct(self,
                  distances: Tensor = 1,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  atom_mask: Tensor = None,
                  ):

        rbf = self.get_rbf(distances)
        rbf_self = 0
        if self.r_self is not None:
            self.get_rbf(self.r_self_ex)

        # apply cutoff
        if self.cutoff_fn is None:
            cutoff = F.ones_like(neighbours)
        else:
            cutoff, neighbour_mask = self.cutoff_fn(neighbours, neighbour_mask)

        cutoff_self = 1
        if self.r_self is not None:
            cutoff_self = F.cast(atom_mask, ms.float32)

        return rbf, cutoff, neighbour_mask, rbf_self, cutoff_self
