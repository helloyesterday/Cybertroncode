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
from mindspore.nn import Cell, get_activation
from mindspore.ops import functional as F
from mindspore.common.initializer import Initializer, Normal


from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length
from mindsponge.function import get_integer, get_ms_array, get_arguments, get_initializer

from .cutoff import Cutoff, get_cutoff
from .rbf import get_rbf
from .filter import Filter, get_filter

__all__ = [
    'GraphEmbedding',
    'MolEmbedding',
    'ConformationEmbedding',
]

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
    def __init__(self,
                 dim_node: int,
                 dim_edge: int,
                 activation: Cell = None,
                 length_unit: Union[str, Units] = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self._dim_node = get_integer(dim_node)
        self._dim_edge = get_integer(dim_edge)

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


@_embedding_register('molecule')
class MolEmbedding(GraphEmbedding):
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
                 dis_filter: Union[Filter, str] = 'residual',
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
            dim_edge=(dim_node if dim_edge is None else dim_edge),
            activation=activation,
            length_unit=length_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.emb_dis = emb_dis
        self.emb_bond = emb_bond

        self.num_atom_types = get_integer(num_atom_types)
        self.initializer = get_initializer(initializer)
        self.atom_embedding = None
        self.atom_embedding = nn.Embedding(vocab_size=self.num_atom_types,
                                           embedding_size=self.dim_node,
                                           use_one_hot=True,
                                           embedding_table=self.initializer)

        cutoff = get_length(cutoff, self.units)
        self.cutoff_fn = get_cutoff(cutoff_fn, cutoff)
        if self.cutoff_fn is None:
            self.cutoff = get_ms_array(cutoff, ms.float32)
        else:
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
        if self.atom_filter is not None:
            node_emb = self.atom_filter(node_emb)

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


@_embedding_register('conformation')
class ConformationEmbedding(MolEmbedding):
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


_EMBEDDING_BY_NAME = {filter.__name__: filter for filter in _EMBEDDING_BY_KEY.values()}


def get_embedding(cls_name: Union[GraphEmbedding, str],
                  dim_node: int,
                  dim_edge: int,
                  activation: Cell = None,
                  length_unit: str = GLOBAL_UNITS.length_unit,
                  **kwargs,
                  ) -> GraphEmbedding:
    """get graph embedding by name"""

    if isinstance(cls_name, GraphEmbedding):
        return cls_name

    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_embedding(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _EMBEDDING_BY_KEY.keys():
            return _EMBEDDING_BY_KEY[cls_name.lower()](dim_node=dim_node,
                                                       dim_edge=dim_edge,
                                                       activation=activation,
                                                       length_unit=length_unit,
                                                       **kwargs
                                                       )
        if cls_name in _EMBEDDING_BY_NAME.keys():
            return _EMBEDDING_BY_NAME[cls_name](dim_node=dim_node,
                                                dim_edge=dim_edge,
                                                activation=activation,
                                                length_unit=length_unit,
                                                **kwargs
                                                )

        raise ValueError(
            "The filter corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported filter type '{}'.".format(type(cls_name)))
