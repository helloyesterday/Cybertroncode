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
from mindspore import Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from mindsponge.function import get_integer, gather_vector, get_arguments

from .interaction import Interaction, _interaction_register
from ..block import Dense
from ..block import PreActDense
from ..block import SeqPreActResidual


@_interaction_register('physnet')
class PhysNetModule(Interaction):
    r"""PhysNet Module (Interaction layer)

    Args:

        dim_feature (int):          Feature dimension.

        activation (Cell):          Activation function. Default: 'silu'

        n_inter_residual (int):     Number of inter residual blocks. Default: 3

        n_outer_residual (int):     Number of outer residual blocks. Default: 2

    """
    def __init__(self,
                 dim_feature: int,
                 dim_edge_emb: int = None,
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
                 activation: Union[Cell, str] = 'silu',
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

        dim_feature = self.dim_node_rep

        self.xi_dense = Dense(
            dim_feature, dim_feature, activation=self.activation)
        self.xij_dense = Dense(
            dim_feature, dim_feature, activation=self.activation)

        self.attention_mask = Dense(self.dim_edge_emb, self.dim_edge_rep,
                                    has_bias=False, activation=None)

        self.gating_vector = Parameter(initializer(Normal(1.0), [dim_feature]),
                                       name="gating_vector")

        self.n_inter_residual = get_integer(n_inter_residual)
        self.n_outer_residual = get_integer(n_outer_residual)

        self.inter_residual = SeqPreActResidual(dim_feature, activation=self.activation,
                                                n_res=self.n_inter_residual)
        self.inter_dense = PreActDense(dim_feature, dim_feature, activation=self.activation)
        self.outer_residual = SeqPreActResidual(dim_feature, activation=self.activation,
                                                n_res=self.n_outer_residual)

        self.reducesum = P.ReduceSum()

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + self.dim_node_rep)
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Number of layers at inter residual: ' +
              self.n_inter_residual)
        print(ret+gap+' Number of layers at outer residual: ' +
              self.n_outer_residual)
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

        xi = self.activation(node_vec)
        xij = gather_vector(xi, neigh_list)

        ux = self.gating_vector * node_vec

        dxi = self.xi_dense(xi)
        dxij = self.xij_dense(xij)

        attention_mask = self.attention_mask(edge_vec * F.expand_dims(edge_cutoff, -1))

        side = attention_mask * dxij
        if edge_mask is not None:
            side = side * F.expand_dims(edge_mask, -1)
        v = dxi + self.reducesum(side, -2)

        v1 = self.inter_residual(v)
        v1 = self.inter_dense(v1)
        y = ux + v1

        node_new = self.outer_residual(y)

        return node_new, edge_vec
