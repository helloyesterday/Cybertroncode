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

from typing import Union, List

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore.common.initializer import Initializer, Normal

from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length
from mindsponge.function import get_integer, get_ms_array

from .cutoff import Cutoff, get_cutoff
from .rbf import get_rbf
from .filter import Filter, get_filter

__all__ = [
    'GraphEmbedding',
]

class GraphEmbedding(nn.Cell):
    def __init__(self,
                 dim_node: int = 128,
                 dim_edge: int = 128,
                 emb_dis: bool = True,
                 emb_bond: bool = False,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf_fn: Cell = None,
                 num_basis: int = None,
                 dis_filter: Union[Filter, str] = 'residual',
                 bond_filter: Union[Filter, str] = 'residual',
                 interaction: Cell = None,
                 dis_self: Length = Length(0.05, 'nm'),
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 initializer: Union[Initializer, str] = Normal(1.0),
                 activation: Cell = 'swish',
                 length_unit: str = 'nm',
                 ):

        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.emb_dis = emb_dis
        self.emb_bond = emb_bond

        num_atom_types = get_integer(num_atom_types)
        self._dim_node = get_integer(dim_node)
        self._dim_edge = get_integer(dim_edge)
        self.atom_embedding = nn.Embedding(
            num_atom_types, self._dim_node,
            use_one_hot=True, embedding_table=initializer)

        cutoff = get_length(cutoff, self.units)
        self.cutoff = get_ms_array(cutoff, ms.float32)
        self.cutoff_fn = get_cutoff(cutoff_fn, self.cutoff)

        dis_self = get_length(dis_self, self.units)
        # (1)
        self.dis_self = get_ms_array(dis_self, ms.float32).reshape((-1,))

        self.num_bond_types = get_integer(num_bond_types)

        self.rbf_fn = None
        self.dis_filter = None
        if self.emb_dis:
            num_basis = get_integer(num_basis)
            self.rbf_fn = get_rbf(rbf_fn, r_max=self.cutoff, num_basis=num_basis,
                                length_unit=self.units.length_unit)

            self.dis_filter = get_filter(filter=dis_filter,
                                        dim_in=self.num_basis,
                                        dim_out=self._dim_edge,
                                        activation=activation)

        self.bond_embedding = None
        self.bond_filter = None
        if self.emb_bond:
            self.bond_embedding = nn.Embedding(
                self.num_bond_types, self._dim_edge,
                use_one_hot=True, embedding_table=initializer)
            self.bond_filter = get_filter(filter=bond_filter,
                                        dim_in=self._dim_edge,
                                        dim_out=self._dim_edge,
                                        activation=activation)
        
        if self.emb_dis and self.emb_bond:
            self.interaction = interaction

    @property
    def num_basis(self) -> int:
        if self.rbf_fn is None:
            return 1
        return self.rbf_fn.num_basis

    @property
    def dim_node(self) -> int:
        return self._dim_node

    @property
    def dim_edge(self) -> int:
        if self.emb_dis and self.dis_filter is None:
            return self.num_basis
        return self._dim_edge

    def get_rbf(self, distances: Tensor):
        """get radical basis function"""
        if self.rbf_fn is None:
            # (B, A, N, 1)
            return F.expand_dims(distances, -1)
        # (B, A, N, F)
        return self.rbf_fn(distances)

    def construct(self,
                  atom_types: Tensor,
                  atom_mask: Tensor,
                  neigh_list: Tensor,
                  distance: Tensor,
                  dis_mask: Tensor,
                  bond: Tensor,
                  bond_mask: Tensor,
                  **kwargs,
                  ):
        
        if self.emb_dis:
            batch_size = distance.shape[0]
            num_atoms = distance.shape[-2]
        else:
            batch_size = bond.shape[0]
            num_atoms = bond.shape[-2]

        node_emb = self.atom_embedding(atom_types)
        node_mask = atom_mask
        if batch_size > 1 and atom_types.shape[0] != batch_size:
            node_emb = msnp.broadcast_to(node_emb, (batch_size,) + node_emb.shape[1:])
            if atom_mask is not None:
                node_mask = msnp.broadcast_to(atom_mask, (batch_size,)+atom_mask.shape[1:])

        dis_emb = None
        dis_mask = None
        dis_cutoff = None
        dis_self = None
        if self.emb_dis:
            # (B, A, N, K)
            dis_emb = self.get_rbf(distance)
            # (1, K)
            dis_self = self.get_rbf(self.dis_self)
            if self.dis_filter is not None:
                # (B, A, N, F)
                dis_emb = self.dis_filter(dis_emb)
                # (1, F)
                dis_self = self.dis_filter(dis_self)

            # (B, A, N)
            if self.cutoff_fn is None:
                dis_cutoff = F.ones_like(distance)
            else:
                dis_cutoff, dis_mask = self.cutoff_fn(distance, dis_mask)

        bond_emb = None
        bond_mask = None
        bond_cutoff = None
        bond_self = None
        if self.emb_bond:
            bond_emb = self.bond_embedding(bond)
            bond_self = self.bond_embedding(F.zeros((batch_size, num_atoms), ms.int32))

            if bond_mask is not None:
                bond_emb = bond_emb * F.expand_dims(bond_mask, -1)
                bond_cutoff = F.cast(bond_mask > 0, ms.float32)

            if self.bond_filter is not None:
                bond_emb = self.bond_filter(bond_emb)
                bond_self = self.bond_filter(bond_self)

        edge_cutoff = dis_cutoff
        edge_mask = dis_mask
        edge_self = dis_self
        if not self.emb_dis:
            edge_emb = bond_emb
            edge_mask = bond_mask
            edge_cutoff = bond_cutoff
            edge_self = bond_self
        elif not self.emb_bond:
            edge_emb = dis_emb
        else:
            node_emb, edge_emb = self.interaction(node_emb, neigh_list, bond_emb,
                                                  bond_mask, bond_cutoff, bond_self)

        return node_emb, node_mask, edge_emb, edge_mask, edge_cutoff, edge_self
