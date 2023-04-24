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
Deep molecular model
"""

from typing import Union, List

import mindspore as ms
from mindspore.nn import Cell, CellList
from mindspore import Tensor
from mindspore.ops import functional as F

from mindsponge.function import concat_last_dim
from mindsponge.function import get_integer, get_ms_array, get_arguments

from .model import MolecularGNN, _model_register
from ..interaction import Interaction, NeuralInteractionUnit
from ..filter import ResFilter


@_model_register('molct')
class MolCT(MolecularGNN):
    r"""Molecular Configuration Transformer (MolCT) Model

    Reference:

        Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
        Molecular CT: unifying geometry and representation learning for molecules at different scales [J/OL].
        arXiv preprint, 2020: arXiv:2012.11816 [2020-12-22]. https://arxiv.org/abs/2012.11816

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 3

        n_heads (int):              Number of heads in multi-head attention. Default: 8

        max_cycles (int):           Maximum number of cycles of the adapative computation time (ACT).
                                    Default: 10

        activation (Cell):          Activation function. Default: 'silu'

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        fixed_cycles (bool):        Whether to use the fixed cycle number to do ACT. Default: False

        use_feed_forward (bool):    Whether to use feed forward after multi-head attention. Default: False

        act_threshold (float):      Threshold of adapative computation time. Default: 0.9

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 128,
                 dim_edge_emb: int = None,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 3,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 activation: Union[Cell, str] = 'silu',
                 coupled_interaction: bool = False,
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 act_threshold: float = 0.9,
                 **kwargs
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            n_interaction=n_interaction,
            interaction=interaction,
            activation=activation,
            coupled_interaction=coupled_interaction,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.n_heads = get_integer(n_heads)
        self.max_cycles = get_integer(max_cycles)
        self.use_feed_forward = use_feed_forward
        self.fixed_cycles = fixed_cycles
        self.act_threshold = get_ms_array(act_threshold, ms.float32)

        self.dim_feature = get_integer(dim_feature)

        self.filter_net = None
        if self.dim_edge_emb is not None:
            self.filter_net = ResFilter(self.dim_edge_emb, self.dim_feature, self.activation)
            if self.interaction is None:
                self.build_interaction()

        self.default_embedding = self.get_default_embedding('molct')

    def build_interaction(self):
        if self.dim_edge_emb is None:
            raise ValueError('Cannot build interaction without `dim_edge_emb`. '
                             'Please use `set_embedding_dimension` at first.')

        if self.coupled_interaction:
            self.interaction = CellList(
                [
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        fixed_cycles=self.fixed_cycles,
                        use_feed_forward=self.use_feed_forward,
                        act_threshold=self.act_threshold,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction = CellList(
                [
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        fixed_cycles=self.fixed_cycles,
                        use_feed_forward=self.use_feed_forward,
                        act_threshold=self.act_threshold,
                    )
                    for _ in range(self.n_interaction)
                ]
            )

    def set_dimension(self, dim_node_emb: int, dim_edge_emb: int):
        """check and set dimension of embedding vectors"""
        super().set_dimension(dim_node_emb, dim_edge_emb)
        if self.filter_net is None:
            self.filter_net = ResFilter(self.dim_edge_emb, self.dim_feature, self.activation)
        return self

    def construct(self,
                  node_emb: Tensor,
                  node_mask: Tensor = None,
                  neigh_list: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_mask: Tensor = None,
                  edge_cutoff: Tensor = None,
                  edge_self: Tensor = None,
                  **kwargs
                  ):
        """Compute the representation of atoms.

        Args:

            node_emb (Tensor):    Tensor of shape (B, A, F). Data type is float
                                        Atom embedding.
            distances (Tensor):         Tensor of shape (B, A, N). Data type is float
                                        Distances between atoms.
                                        Atomic number.
            atom_mask (Tensor):         Tensor of shape (B, A). Data type is bool
                                        Mask of atomic number
            neighbours (Tensor):        Tensor of shape (B, A, N). Data type is int
                                        Neighbour index.
            neighbour_mask (Tensor):    Tensor of shape (B, A, N). Data type is bool
                                        Nask of neighbour index.

        Returns:
            representation: (Tensor)    Tensor of shape (B, A, F). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates, usually is 3.
            F:  Feature dimension of representation.

        """

        # (B,A) -> (B,A,1)
        node_mask = F.expand_dims(node_mask, -1)

        c_ii = F.cast(node_mask, ms.float32)
        edge_cutoff = concat_last_dim((c_ii, edge_cutoff))
        edge_mask = concat_last_dim((node_mask, edge_mask))

        node_vec = node_emb
        edge_vec = self.filter_net(edge_emb)
        edge_self = self.filter_net(edge_self)
        for i in range(len(self.interaction)):
            node_vec, edge_vec = self.interaction[i](
                node_vec=node_vec,
                edge_vec=edge_vec,
                edge_cutoff=edge_cutoff,
                neigh_list=neigh_list,
                edge_mask=edge_mask,
                node_emb=node_emb,
                edge_self=edge_self,
            )

        return node_vec, edge_vec
