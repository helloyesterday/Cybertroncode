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

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F

from mindsponge.function import get_integer, gather_vector, get_arguments
from mindsponge.function import concat_last_dim, squeeze_penulti

from .interaction import Interaction, _interaction_register
from ..base import PositionalEmbedding
from ..base import MultiheadAttention
from ..base import Pondering, ACTWeight
from ..base import FeedForward


@_interaction_register('niu')
class NeuralInteractionUnit(Interaction):
    r"""Neural interaction unit for MolCT.

    Args:

        dim_feature (int):          Feature dimension.

        n_heads (int):              Number of head for multi-head attention. Default: 8

        max_cycles (int):           Maximum cycles for adaptive computation time (ACT). Default: 10

        activation (Cell):          Activation function. Default: 'silu'

        filter_net (Cell):          Filter network for edge vector. Default: None

        fixed_cycles (bool):        Whether to fixed number of cyles to do ACT. Default: False

        use_feed_forward (bool):    Whether to use feed forward network. Default: False

        act_threshold (float):      Threshold value for ACT. Default: 0.9


    """

    def __init__(self,
                 dim_feature: int,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 activation: Union[Cell, str] = 'silu',
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 act_threshold: float = 0.9,
                 **kwargs,
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_feature,
            activation=activation,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.n_heads = get_integer(n_heads)
        self.dim_feature = get_integer(dim_feature)

        if self.dim_feature % self.n_heads != 0:
            raise ValueError('The term "dim_feature" cannot be divisible ' +
                             'by the term "n_heads" in AirNetIneteraction! ')

        self.max_cycles = get_integer(max_cycles)

        self.fixed_cycles = fixed_cycles

        if self.fixed_cycles:
            self.time_embedding = [0 for _ in range(self.max_cycles)]
        else:
            self.time_embedding = self.get_time_signal(
                self.max_cycles, dim_feature)

        self.positional_embedding = PositionalEmbedding(dim_feature)
        self.multi_head_attention = MultiheadAttention(dim_feature, self.n_heads, dim_tensor=4)

        self.use_feed_forward = use_feed_forward
        self.feed_forward = None
        if self.use_feed_forward:
            self.feed_forward = FeedForward(dim_feature, self.activation)

        self.act_threshold = act_threshold
        self.act_epsilon = 1.0 - act_threshold

        self.pondering = None
        self.act_weight = None
        self.open_act = False
        if self.max_cycles > 1:
            self.open_act = True
            self.pondering = Pondering(dim_feature*3, bias_const=3)
            self.act_weight = ACTWeight(self.act_threshold)

    @staticmethod
    def get_time_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4) -> Tensor:
        """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py
        """
        position = msnp.arange(length, dtype=ms.float32)
        num_timescales = channels // 2
        log_timescale_increment = msnp.log(
            max_timescale / min_timescale, dtype=ms.float32) / (num_timescales - 1)
        inv_timescales = min_timescale * \
            msnp.exp(msnp.arange(num_timescales, dtype=ms.float32)
                     * -log_timescale_increment)
        scaled_time = F.expand_dims(position, 1) * \
            F.expand_dims(inv_timescales, 0)

        signal = msnp.concatenate([msnp.sin(scaled_time, dtype=ms.float32), msnp.cos(
            scaled_time, dtype=ms.float32)], axis=1)
        signal = msnp.pad(signal, [[0, 0], [0, channels % 2]],
                          'constant', constant_values=[0.0, 0.0])

        return signal

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_node_rep))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Number of heads in multi-haed attention: '+str(self.n_heads))
        print(ret+gap+' Use feed forward network: ' +
              ('Yes' if self.use_feed_forward else 'No'))
        if self.max_cycles > 1:
            print(
                ret+gap+' Adaptive computation time (ACT) with maximum cycles: '+str(self.max_cycles))
            print(ret+gap+' Cycle mode: ' +
                  ('Fixed' if self.fixed_cycles else 'Fixible'))
            print(ret+gap+' Threshold for ACT: '+str(self.act_threshold))
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

        def _encoder(node_vec: Tensor,
                     neigh_list: Tensor,
                     edge_vec: Tensor = 1,
                     edge_mask: Tensor = None,
                     edge_cutoff: Tensor = None,
                     edge_self: Tensor = 1,
                     time_signal: Tensor = 0,
                     ) -> Tensor:

            """encoder for transformer"""

            # (B, A, N, F) <- (B, A, F)
            node_mat = gather_vector(node_vec, neigh_list)
            query, key, value = self.positional_embedding(
                node_vec, node_mat, edge_self, edge_vec, time_signal)
            dv = self.multi_head_attention(query, key, value, mask=edge_mask, cutoff=edge_cutoff)
            dv = squeeze_penulti(dv)

            node_new = node_vec + dv

            if self.use_feed_forward:
                node_new = self.feed_forward(node_new)

            return node_new

        if self.open_act:
            def _act_encoder(node_new: Tensor,
                             node_vec: Tensor,
                             node_emb: Tensor,
                             neigh_list: Tensor,
                             edge_vec: Tensor,
                             edge_mask: Tensor,
                             edge_cutoff: Tensor,
                             edge_self: Tensor,
                             halting_prob: Tensor,
                             n_updates: Tensor,
                             cycle: int,
                             ) -> Tensor:

                time_signal = self.time_embedding[cycle]
                vt = F.zeros_like(node_emb) + time_signal

                xp = concat_last_dim((node_vec, node_emb, vt))
                p = self.pondering(xp)
                w, dp, dn = self.act_weight(p, halting_prob)
                halting_prob = halting_prob + dp
                n_updates = n_updates + dn

                node_vec = _encoder(
                    node_vec=node_vec,
                    neigh_list=neigh_list,
                    edge_vec=edge_vec,
                    edge_mask=edge_mask,
                    edge_cutoff=edge_cutoff,
                    edge_self=edge_self,
                    time_signal=time_signal
                    )

                node_new = node_vec * w + node_new * (1.0 - w)

                return node_new, node_vec, halting_prob, n_updates

            node_new = F.zeros_like(node_vec)

            halting_prob = F.zeros((node_vec.shape[0], node_vec.shape[1]), ms.float32)
            n_updates = F.zeros((node_vec.shape[0], node_vec.shape[1]), ms.float32)

            if self.fixed_cycles:
                for cycle in range(self.max_cycles):
                    node_new, node_vec, halting_prob, n_updates = _act_encoder(
                        node_new=node_new,
                        node_vec=node_vec,
                        node_emb=node_emb,
                        neigh_list=neigh_list,
                        edge_vec=edge_vec,
                        edge_mask=edge_mask,
                        edge_cutoff=edge_cutoff,
                        edge_self=edge_self,
                        halting_prob=halting_prob,
                        n_updates=n_updates,
                        cycle=cycle,
                        )
            else:
                cycle = F.zeros((), ms.int32)
                while((halting_prob < self.act_threshold).any() and (cycle < self.max_cycles)):
                    node_new, node_vec, halting_prob, n_updates = _act_encoder(
                        node_new=node_new,
                        node_vec=node_vec,
                        node_emb=node_emb,
                        neigh_list=neigh_list,
                        edge_vec=edge_vec,
                        edge_mask=edge_mask,
                        edge_cutoff=edge_cutoff,
                        edge_self=edge_self,
                        halting_prob=halting_prob,
                        n_updates=n_updates,
                        cycle=cycle,
                        )
                    cycle = cycle + 1
        else:
            time_signal = self.time_embedding[0]
            node_new = _encoder(node_vec=node_vec,
                                neigh_list=neigh_list,
                                edge_vec=edge_vec,
                                edge_mask=edge_mask,
                                edge_cutoff=edge_cutoff,
                                edge_self=edge_self,
                                time_signal=time_signal
                                )

        return node_new, edge_vec
