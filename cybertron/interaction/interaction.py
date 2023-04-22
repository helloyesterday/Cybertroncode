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
Interaction layers
"""

from typing import Union

from mindspore import Tensor
from mindspore.nn import Cell

from mindsponge.function import get_integer

from ..activation import get_activation

_INTERACTION_BY_KEY = dict()


def _interaction_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _INTERACTION_BY_KEY:
            _INTERACTION_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _INTERACTION_BY_KEY:
                _INTERACTION_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Interaction(Cell):
    r"""Interaction layer network

    Args:

        dim_feature (int):          Feature dimension.

        activation (Cell):          Activation function. Default: None

        use_distance (bool):        Whether to use distance between atoms. Default: True

        use_bond (bool):            Whether to use bond information. Default: False


    """

    def __init__(self,
                 dim_node_rep: int,
                 dim_edge_rep: int,
                 dim_node_emb: int = None,
                 dim_edge_emb: int = None,
                 activation: Union[Cell, str] = 'silu',
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        self.dim_node_rep = get_integer(dim_node_rep)
        self.dim_edge_rep = get_integer(dim_edge_rep)
        self.dim_node_emb = get_integer(dim_node_emb)
        self.dim_edge_emb = get_integer(dim_edge_emb)
        if self.dim_edge_emb is None:
            self.dim_edge_emb = self.dim_node_rep
        self.activation = get_activation(activation)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        """print information of interaction layer"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_node_rep))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print('-'*80)
        return self

    def construct(self,
                  node_vec: Tensor,
                  node_emb: Tensor,
                  neigh_list: Tensor,
                  edge_vec: Tensor,
                  edge_mask: Tensor = None,
                  edge_cutoff: Tensor = None,
                  edge_self: Tensor = None,
                  **kwargs
                  ):

        """Compute interaction layer.

        Args:
            x (Tensor):             Tensor of shape (B, A, F). Data type is float
                                    Representation of each atom.
            f_ij (Tensor):          Tensor of shape (B, A, N, F). Data type is float
                                    Edge vector of distance.
            bond_vec (Tensor):          Tensor of shape (B, A, N, F). Data type is float
                                    Edge vector of bond connection.
            c_ij (Tensor):          Tensor of shape (B, A, N). Data type is float
                                    Cutoff for distance.
            neighbours (Tensor):    Tensor of shape (B, A, N). Data type is int
                                    Neighbour index.
            mask (Tensor):          Tensor of shape (B, A, N). Data type is bool
                                    Mask of neighbour index.
            node_emb (Tensor):             Tensor of shape (B, A, F). Data type is float
                                    Embdding vector for each atom
            edge_self (Tensor):          Tensor of shape (B, A, 1, F). Data type is float
                                    Edge vector of distance for atom itself.
            bond_self (Tensor):          Tensor of shape (B, A, 1, F). Data type is float
                                    Edge vector of bond connection for atom itself.
            cutoff_self (Tensor):          Tensor of shape (B, A). Data type is float
                                    Cutoff for atom itself.
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool
                                    Mask for each atom

        Returns:
            y: (Tensor)             Tensor of shape (B, A, F). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates, usually is 3.
            F:  Feature dimension of representation.

        """

        #pylint: disable=unused-argument
        return node_vec, edge_vec
