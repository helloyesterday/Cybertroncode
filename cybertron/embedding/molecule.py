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
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore.common.initializer import Initializer, Normal


from mindsponge.function import GLOBAL_UNITS, Length, get_length
from mindsponge.function import get_integer, get_ms_array, get_arguments, get_initializer

from .graph import GraphEmbedding, _embedding_register
from ..cutoff import Cutoff, get_cutoff
from ..rbf import get_rbf
from ..filter import Filter, get_filter


@_embedding_register('molecule')
class MolEmbedding(GraphEmbedding):
    r"""Embedding for molecule

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off distance. Default: None

        activation: Union[Cell, str]: Activation function. Default: None

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    """
    def __init__(self,
                 dim_node: int,
                 dim_edge: int = None,
                 emb_dis: bool = True,
                 emb_bond: bool = False,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = 'smooth',
                 rbf_fn: Cell = 'log_gaussian',
                 num_basis: int = None,
                 atom_filter: Union[Filter, str] = None,
                 dis_filter: Union[Filter, str] = None,
                 bond_filter: Union[Filter, str] = 'residual',
                 interaction: Cell = None,
                 dis_self: Length = Length(0.05, 'nm'),
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 initializer: Union[Initializer, str] = Normal(1.0),
                 activation: Cell = 'silu',
                 length_unit: str = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):

        super().__init__(
            dim_node=dim_node,
            dim_edge=dim_node if dim_edge is None else dim_edge,
            emb_dis=emb_dis,
            emb_bond=emb_bond,
            cutoff=cutoff,
            activation=activation,
            length_unit=length_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.num_atom_types = get_integer(num_atom_types)
        self.initializer = get_initializer(initializer)
        self.atom_embedding = None
        self.atom_embedding = nn.Embedding(vocab_size=self.num_atom_types,
                                           embedding_size=self.dim_node,
                                           use_one_hot=True,
                                           embedding_table=self.initializer)

        self.cutoff_fn = get_cutoff(cutoff_fn, cutoff)
        if self.cutoff_fn is not None:
            self.cutoff = self.cutoff_fn.cutoff

        dis_self = get_length(dis_self, self.units)
        # (1)
        self.dis_self = get_ms_array(dis_self, ms.float32).reshape((-1,))

        self.num_bond_types = get_integer(num_bond_types)

        self.atom_filter = get_filter(atom_filter, self.dim_node,
                                      self.dim_node, activation)

        self.rbf_fn = None
        self.dis_filter = None
        if self.emb_dis:
            num_basis = get_integer(num_basis)
            self.rbf_fn = get_rbf(rbf_fn, r_max=self.cutoff, num_basis=num_basis,
                                  length_unit=self.units.length_unit)

            self.dis_filter = get_filter(cls_name=dis_filter,
                                         dim_in=self.num_basis,
                                         dim_out=self._dim_edge,
                                         activation=activation)

        self.bond_embedding = None
        self.bond_filter = None
        if self.emb_bond:
            self.bond_embedding = nn.Embedding(
                self.num_bond_types, self._dim_edge,
                use_one_hot=True, embedding_table=initializer)
            self.bond_filter = get_filter(cls_name=bond_filter,
                                          dim_in=self._dim_edge,
                                          dim_out=self._dim_edge,
                                          activation=activation)

        if self.emb_dis and self.emb_bond:
            self.interaction = interaction

    @property
    def num_basis(self) -> int:
        """number of radical basis function"""
        if self.rbf_fn is None:
            return 1
        return self.rbf_fn.num_basis

    @property
    def dim_edge(self) -> int:
        """dimension of edge vector"""
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
        #pylint: disable=unused-argument

        if self.emb_dis:
            batch_size = neigh_dis.shape[0]
            num_atoms = neigh_dis.shape[-2]
        else:
            batch_size = bond.shape[0]
            num_atoms = bond.shape[-2]

        node_emb = self.atom_embedding(atom_type)
        if self.atom_filter is not None:
            node_emb = self.atom_filter(node_emb)

        node_mask = atom_mask
        if batch_size > 1 and atom_type.shape[0] != batch_size:
            node_emb = msnp.broadcast_to(node_emb, (batch_size,) + node_emb.shape[1:])
            if atom_mask is not None:
                node_mask = msnp.broadcast_to(atom_mask, (batch_size,)+atom_mask.shape[1:])

        dis_emb = None
        neigh_mask = None
        dis_cutoff = None
        dis_self = None
        if self.emb_dis:
            # (B, A, N, K)
            dis_emb = self.get_rbf(neigh_dis)
            # (1, K)
            dis_self = self.get_rbf(self.dis_self)
            if self.dis_filter is not None:
                # (B, A, N, F)
                dis_emb = self.dis_filter(dis_emb)
                # (1, F)
                dis_self = self.dis_filter(dis_self)

            # (B, A, N)
            if self.cutoff_fn is None:
                dis_cutoff = F.ones_like(neigh_dis)
            else:
                dis_cutoff, neigh_mask = self.cutoff_fn(neigh_dis, neigh_mask)

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
        edge_mask = neigh_mask
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


@_embedding_register('conformation')
class ConformationEmbedding(MolEmbedding):
    r"""Embedding for molecular conformation

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off distance. Default: None

        activation: Union[Cell, str]: Activation function. Default: None

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    """
    def __init__(self,
                 dim_feature: int,
                 emb_bond: bool = False,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf_fn: Cell = None,
                 num_basis: int = None,
                 atom_filter: Union[Filter, str] = None,
                 dis_filter: Union[Filter, str] = 'residual',
                 bond_filter: Union[Filter, str] = 'residual',
                 interaction: Cell = None,
                 dis_self: Length = Length(0.05, 'nm'),
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 initializer: Union[Initializer, str] = Normal(1.0),
                 activation: Cell = 'swish',
                 length_unit: str = None,
                 **kwargs,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            emb_dis=True,
            emb_bond=emb_bond,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf_fn=rbf_fn,
            num_basis=num_basis,
            atom_filter=atom_filter,
            dis_filter=dis_filter,
            bond_filter=bond_filter,
            interaction=interaction,
            dis_self=dis_self,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            initializer=initializer,
            activation=activation,
            length_unit=length_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)
