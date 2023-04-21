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
from mindspore.ops import functional as F

from mindsponge.function import get_integer, gather_vector, get_arguments

from .interaction import Interaction, _interaction_register
from ..block import Dense, MLP
from ..base import Aggregate
from ..filter import DenseFilter


@_interaction_register('schnet')
class SchNetInteraction(Interaction):
    r"""Interaction layer of SchNet.

    Args:

        dim_feature (int):          Feature dimension.

        dim_filter (int):           Dimension of filter network.

        filter_net (Cell):          Filter network for distance

        activation (Cell):          Activation function. Default: 'silu'

        normalize_filter (bool):    Whether to nomalize filter network. Default: False

    """

    def __init__(self,
                 dim_feature: int,
                 dim_edge_emb: int,
                 dim_filter: int,
                 activation: Union[Cell, str] = 'silu',
                 normalize_filter: bool = False,
                 **kwargs,
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
            activation=activation,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.dim_filter = get_integer(dim_filter)
        if dim_filter is None:
            self.dim_filter = self.dim_edge_rep

        # (..., K) -> (..., W)
        self.filter_net = DenseFilter(dim_in=self.dim_edge_emb, dim_out=self.dim_filter,
                                      activation=activation)

        # (..., F) -> (..., W)
        self.atomwise_bc = Dense(self.dim_node_emb, self.dim_filter)
        # (..., W) -> (..., F)
        self.atomwise_ac = MLP(self.dim_filter, self.dim_node_rep, [self.dim_node_rep],
                               activation=activation, use_last_activation=False)

        self.agg = Aggregate(axis=-2, mean=normalize_filter)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
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

        # (B, A, W) <- (B, A, F)
        ax = self.atomwise_bc(node_vec)
        # (B, A, N, W)
        x_ij = gather_vector(ax, neigh_list)

        # (B, A, N, K)
        g_ij = edge_vec
        # (B, A, N, W) <- (B, A, N, K)
        g_ij = self.filter_net(edge_vec)

        # CFconv: pass expanded interactomic distances through filter block
        # (B, A, N, W) * (B, A, N, 1)
        w_ij = g_ij * F.expand_dims(edge_cutoff, -1)
        # (B, A, N, W) * (B, A, N, W)
        y = x_ij * w_ij

        # atom-wise multiplication, aggregating and Dense layer
        # (B, A, W) <- (B, A, N, W)
        y = self.agg(y, edge_mask)
        # (B, A, F) <- (B, A, W)
        v = self.atomwise_ac(y)

        # (B, A, F) + (B, A, F)
        node_new = node_vec + v

        return node_new, edge_vec
