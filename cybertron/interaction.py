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
from mindspore import Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

import mindsponge.function as func

from .block import Dense, MLP
from .block import PreActDense
from .block import SeqPreActResidual
from .base import Aggregate
from .base import PositionalEmbedding
from .base import MultiheadAttention
from .base import Pondering, ACTWeight
from .base import FeedForward
from .activation import get_activation
from .filter import Filter, get_filter

__all__ = [
    "Interaction",
    "SchNetInteraction",
    "PhysNetModule",
    "NeuralInteractionUnit",
]


class Interaction(Cell):
    r"""Interaction layer network

    Args:

        dim_feature (int):          Feature dimension.

        activation (Cell):          Activation function. Default: None

        use_distance (bool):        Whether to use distance between atoms. Default: True

        use_bond (bool):            Whether to use bond information. Default: False


    """

    def __init__(self,
                 dim_feature: int,
                 dim_input: int = None,
                 filter_net: Union[Filter, str] = None,
                 filter_layer: int = 1,
                 activation: Cell = None,
                 ):
        super().__init__()

        self.reg_key = 'interaction'
        self.name = 'Interaction'
        self.dim_feature = func.get_integer(dim_feature)
        self.dim_input = func.get_integer(dim_input)
        if self.dim_input is None:
            self.dim_input = self.dim_feature
        self.activation = get_activation(activation)

        filter_layer = func.get_integer(filter_layer)

        self.filter_net = get_filter(filter_net, self.dim_input,
                                     self.dim_feature, self.activation, filter_layer)

        self.gather_neighbours = func.gather_vector

    def set_filter(self, filter_net: Union[Filter, str], filter_layer: int = 1):
        self.filter_net = get_filter(filter_net, self.dim_input,
                                     self.dim_feature, self.activation, filter_layer)
        return self

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        """print information of interaction layer"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print('-'*80)
        return self

    def _output_block(self, x: Tensor) -> Tensor:
        """output block network"""
        return x

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


class SchNetInteraction(Interaction):
    r"""Interaction layer of SchNet.

    Args:

        dim_feature (int):          Feature dimension.

        dim_filter (int):           Dimension of filter network.

        filter_net (Cell):          Filter network for distance

        activation (Cell):          Activation function. Default: 'ssp'

        normalize_filter (bool):    Whether to nomalize filter network. Default: False

    """

    def __init__(self,
                 dim_feature: int,
                 dim_input: int = None,
                 filter_net: Union[Filter, str] = 'dense',
                 filter_layer: int = 1,
                 activation: Cell = 'ssp',
                 normalize_filter: bool = False,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            dim_input=dim_input,
            filter_net=filter_net,
            filter_layer=filter_layer,
            activation=activation,
        )

        self.name = 'SchNet Interaction Layer'

        self.atomwise_bc = Dense(self.dim_feature, self.dim_input)
        self.atomwise_ac = MLP(self.dim_input, self.dim_feature, [self.dim_feature],
                               activation=self.activation, use_last_activation=False)

        self.agg = Aggregate(axis=-2, mean=normalize_filter)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Dimension for filter network: ' + str(self.dim_filter))
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

        ax = self.atomwise_bc(node_vec)
        x_ij = self.gather_neighbours(ax, neigh_list)

        g_ij = edge_vec
        if self.filter_net is not None:
            g_ij = self.filter_net(edge_vec)

        # CFconv: pass expanded interactomic distances through filter block
        w_ij = g_ij * F.expand_dims(edge_cutoff, -1)
        y = x_ij * w_ij

        # atom-wise multiplication, aggregating and Dense layer
        y = self.agg(y, edge_mask)
        v = self.atomwise_ac(y)

        node_new = node_vec + v

        return node_new, edge_vec


class PhysNetModule(Interaction):
    r"""PhysNet Module (Interaction layer)

    Args:

        dim_feature (int):          Feature dimension.

        filter_net (Cell):          Filter network for distance

        activation (Cell):          Activation function. Default: 'swish'

        n_inter_residual (int):     Number of inter residual blocks. Default: 3

        n_outer_residual (int):     Number of outer residual blocks. Default: 2

    """
    def __init__(self,
                 dim_feature: int,
                 dim_input: int = None,
                 filter_net: Union[Filter, str] = 'dense',
                 filter_layer: int = 0,
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
                 activation: Cell = 'swish',
                 ):

        super().__init__(
            dim_feature=dim_feature,
            dim_input=dim_input,
            filter_net=filter_net,
            filter_layer=filter_layer,
            activation=activation,
        )

        self.name = 'PhysNet Module Layer'

        self.xi_dense = Dense(
            self.dim_feature, self.dim_feature, activation=self.activation)
        self.xij_dense = Dense(
            self.dim_feature, self.dim_feature, activation=self.activation)

        self.gating_vector = Parameter(initializer(
            Normal(1.0), [self.dim_feature]), name="gating_vector")

        self.n_inter_residual = func.get_integer(n_inter_residual)
        self.n_outer_residual = func.get_integer(n_outer_residual)

        self.inter_residual = SeqPreActResidual(self.dim_feature, activation=self.activation,
                                                n_res=self.n_inter_residual)
        self.inter_dense = PreActDense(self.dim_feature, self.dim_feature, activation=self.activation)
        self.outer_residual = SeqPreActResidual(self.dim_feature, activation=self.activation,
                                                n_res=self.n_outer_residual)

        self.reducesum = P.ReduceSum()

    def set_filter(self, filter_net: Union[Filter, str], filter_layer: int = 0):
        return super().set_filter(filter_net, filter_layer)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + self.dim_feature)
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Number of layers at inter residual: ' +
              self.n_inter_residual)
        print(ret+gap+' Number of layers at outer residual: ' +
              self.n_outer_residual)
        print('-'*80)
        return self

    def _attention_mask(self, f_ij, c_ij) -> Tensor:
        """attention mask"""
        x = f_ij * F.expand_dims(c_ij, -1)

        if self.filter_net is None:
            return x
        return self.filter_net(x)

    def _interaction_block(self, x, f_ij, c_ij, neighbours, mask) -> Tensor:
        """interaction block"""

        #pylint: disable=invalid-name
        xi = self.activation(x)
        xij = self.gather_neighbours(xi, neighbours)

        ux = self.gating_vector * x

        dxi = self.xi_dense(xi)
        dxij = self.xij_dense(xij)
        Ggij = self._attention_mask(f_ij, c_ij)

        side = Ggij * dxij
        if mask is not None:
            side = side * F.expand_dims(mask, -1)
        v = dxi + self.reducesum(side, -2)

        v1 = self.inter_residual(v)
        v1 = self.inter_dense(v1)
        return ux + v1

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

        x1 = self._interaction_block(node_vec, edge_vec, edge_cutoff, neigh_list, edge_mask)
        node_new = self.outer_residual(x1)

        return node_new, edge_vec


class NeuralInteractionUnit(Interaction):
    r"""Neural interaction unit for MolCT.

    Args:

        dim_feature (int):          Feature dimension.

        n_heads (int):              Number of head for multi-head attention. Default: 8

        max_cycles (int):           Maximum cycles for adaptive computation time (ACT). Default: 10

        activation (Cell):          Activation function. Default: 'swish'

        filter_net (Cell):          Filter network for edge vector. Default: None

        fixed_cycles (bool):        Whether to fixed number of cyles to do ACT. Default: False

        use_feed_forward (bool):    Whether to use feed forward network. Default: False

        act_threshold (float):      Threshold value for ACT. Default: 0.9


    """

    def __init__(self,
                 dim_feature: int,
                 dim_input: int = None,
                 filter_net: Union[Filter, str] = 'residual',
                 filter_layer: int = 1,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 activation: Cell = 'swish',
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 act_threshold: float = 0.9,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            dim_input=dim_input,
            filter_net=filter_net,
            filter_layer=filter_layer,
            activation=activation,
        )

        if dim_feature % n_heads != 0:
            raise ValueError('The term "dim_feature" cannot be divisible ' +
                             'by the term "n_heads" in AirNetIneteraction! ')

        self.name = 'Neural Interaction Unit'

        self.n_heads = func.get_integer(n_heads)
        self.max_cycles = func.get_integer(max_cycles)

        self.fixed_cycles = fixed_cycles

        if self.fixed_cycles:
            self.time_embedding = [0 for _ in range(self.max_cycles)]
        else:
            self.time_embedding = self._get_time_signal(
                self.max_cycles, self.dim_feature)

        self.positional_embedding = PositionalEmbedding(self.dim_feature)
        self.multi_head_attention = MultiheadAttention(
            self.dim_feature, self.n_heads, dim_tensor=4)

        self.use_feed_forward = use_feed_forward
        self.feed_forward = None
        if self.use_feed_forward:
            self.feed_forward = FeedForward(self.dim_feature, self.activation)

        self.act_threshold = act_threshold
        self.act_epsilon = 1.0 - act_threshold

        self.pondering = None
        self.act_weight = None
        self.do_act = False
        if self.max_cycles > 1:
            self.do_act = True
            self.pondering = Pondering(self.dim_feature*3, bias_const=3)
            self.act_weight = ACTWeight(self.act_threshold)

        self.concat = P.Concat(-1)
        self.zeros_like = P.ZerosLike()
        self.zeros = P.Zeros()

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Use filter network for edge vector: ' +
              ("No" if self.filter_net is None else "Yes"))
        print(ret+gap+' Number of heads in multi-haed attention: '+str(self.n_heads))
        print(ret+gap+' Use feed forward network: ' +
              ('Yes' if self.use_feed_forward else 'No'))
        if self.max_cycles > 1:
            print(
                ret+gap+' Adaptive computation time (ACT) with maximum cycles: '+str(self.max_cycles))
            print(ret+gap+' Cycle mode: ' +
                  ('Fixed' if self.fixed_cycles else 'Fixible'))
            print(ret+gap+' Threshold for ACT: '+str(self.act_threshold))
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

        edge_vec0 = edge_vec
        if self.filter_net is not None:
            edge_self = self.filter_net(edge_self)
            edge_vec = self.filter_net(edge_vec)

        if self.do_act:
            xx = node_vec
            node_new = self.zeros_like(node_vec)

            halting_prob = self.zeros((node_vec.shape[0], node_vec.shape[1]), ms.float32)
            n_updates = self.zeros((node_vec.shape[0], node_vec.shape[1]), ms.float32)

            broad_zeros = self.zeros_like(node_emb)

            if self.fixed_cycles:
                for cycle in range(self.max_cycles):
                    time_signal = self.time_embedding[cycle]
                    vt = broad_zeros + time_signal

                    xp = self.concat((xx, node_emb, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._encoder(node_vec=xx,
                                       neigh_list=neigh_list,
                                       edge_vec=edge_vec,
                                       edge_mask=edge_mask,
                                       edge_cutoff=edge_cutoff,
                                       edge_self=edge_self,
                                       time_signal=time_signal)

                    cycle = cycle + 1

                    node_new = xx * w + node_new * (1.0 - w)
            else:
                cycle = self.zeros((), ms.int32)
                while((halting_prob < self.act_threshold).any() and (cycle < self.max_cycles)):
                    time_signal = self.time_embedding[cycle]
                    vt = broad_zeros + time_signal
                    xp = self.concat((xx, node_emb, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._encoder(node_vec=xx,
                                       neigh_list=neigh_list,
                                       edge_vec=edge_vec,
                                       edge_mask=edge_mask,
                                       edge_cutoff=edge_cutoff,
                                       edge_self=edge_self,
                                       time_signal=time_signal
                                       )

                    cycle = cycle + 1

                    node_new = xx * w + node_new * (1.0 - w)
        else:
            time_signal = self.time_embedding[0]
            node_new = self._encoder(node_vec=xx,
                                     neigh_list=neigh_list,
                                     edge_vec=edge_vec,
                                     edge_mask=edge_mask,
                                     edge_cutoff=edge_cutoff,
                                     edge_self=edge_self,
                                     time_signal=time_signal)

        return node_new, edge_vec0

    def _encoder(self,
                 node_vec: Tensor,
                 neigh_list: Tensor,
                 edge_vec: Tensor = 1,
                 edge_mask: Tensor = None,
                 edge_cutoff: Tensor = None,
                 edge_self: Tensor = 1,
                 time_signal: Tensor = 0,
                 ) -> Tensor:

        """encoder for transformer"""

        # (B, A, N, F) <- (B, A, F)
        node_mat = self.gather_neighbours(node_vec, neigh_list)
        query, key, value = self.positional_embedding(
            node_vec, node_mat, edge_self, edge_vec, time_signal)
        dv = self.multi_head_attention(
            query, key, value, mask=edge_mask, cutoff=edge_cutoff)
        dv = F.squeeze(dv, -2)

        node_new = node_vec + dv

        if self.use_feed_forward:
            node_new = self.feed_forward(node_new)

        return node_new

    def _get_time_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4) -> Tensor:
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