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

import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.function import get_integer, get_arguments

from .model import MolecularGNN, _model_register
from ..interaction import Interaction, PhysNetModule


@_model_register('physnet')
class PhysNet(MolecularGNN):
    r"""PhysNet Model

    Reference:

        Unke, O. T. and Meuwly, M.,
        PhysNet: A neural network for predicting energyies, forces, dipole moments, and partial charges [J].
        The Journal of Chemical Theory and Computation, 2019, 15(6): 3678-3693.

    Args:

        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 5

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: False

        use_graph_norm (bool):      Whether to use graph normalization. Default: False

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: False

        n_inter_residual (int):     Number of blocks in the inside pre-activation residual block. Default: 3

        n_outer_residual (int):     Number of blocks in the outside pre-activation residual block. Default: 2

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
                 dim_feature: int = 128,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 5,
                 coupled_interaction: bool = False,
                 dim_edge_emb: int = None,
                 activation: Union[Cell, str] = 'ssp',
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
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

        self.n_inter_residual = get_integer(n_inter_residual)
        self.n_outer_residual = get_integer(n_outer_residual)

        self.dim_feature = get_integer(dim_feature)

        if self.interaction is None and self.dim_edge_emb is not None:
            self.build_interaction()

        self.default_embedding = self.get_default_embedding('physnet')

    def build_interaction(self):
        if self.dim_edge_emb is None:
            raise ValueError('Cannot build interaction without `dim_edge_emb`. '
                             'Please use `set_embedding_dimension` at first.')
        if self.coupled_interaction:
            self.interaction = CellList(
                [
                    PhysNetModule(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                        activation=self.activation,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction = CellList(
                [
                    PhysNetModule(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                        activation=self.activation,
                    )
                    for _ in range(self.n_interaction)
                ]
            )
