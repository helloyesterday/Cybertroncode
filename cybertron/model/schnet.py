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

from mindspore.nn import Cell, CellList
from sponge.function import get_integer, get_arguments

from .model import MolecularGNN, _model_register
from ..interaction import Interaction, SchNetInteraction


@_model_register('schnet')
class SchNet(MolecularGNN):
    r"""SchNet Model.

    Reference:

        Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; Müller, K.-R.
        Schnet - a Deep Learning Architecture for Molecules and Materials [J].
        The Journal of Chemical Physics, 2018, 148(24): 241722.

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 64

        dim_filter (int):           Dimension of filter network. Default: 64

        n_interaction (int):        Number of interaction layers. Default: 3

        activation (Cell):          Activation function. Default: 'ssp'

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: 'cosine'

        rbf (Cell):                 Radical baiss function. Default: 'gaussian'

        normalize_filter (bool):    Whether to normalize the filter network. Default: False

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_graph_norm (bool):      Whether to use graph normalization. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 64,
                 dim_filter: int = 64,
                 dim_edge_emb: int = None,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 3,
                 activation: Union[Cell, str] = 'ssp',
                 normalize_filter: bool = False,
                 coupled_interaction: bool = False,
                 **kwargs,
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            interaction=interaction,
            n_interaction=n_interaction,
            activation=activation,
            coupled_interaction=coupled_interaction,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.dim_feature = get_integer(dim_feature)
        self.dim_filter = get_integer(dim_filter)
        self.normalize_filter = normalize_filter

        if self.interaction is None and self.dim_edge_emb is not None:
            self.build_interaction()

        self.default_embedding = self.get_default_embedding('schnet')

    def build_interaction(self):
        if self.dim_edge_emb is None:
            raise ValueError('Cannot build interaction without `dim_edge_emb`. '
                             'Please use `set_embedding_dimension` at first.')

        if self.coupled_interaction:
            self.interaction = CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        normalize_filter=self.normalize_filter,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction = CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        normalize_filter=self.normalize_filter,
                    )
                    for _ in range(self.n_interaction)
                ]
            )
