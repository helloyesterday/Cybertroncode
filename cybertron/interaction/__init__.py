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
Interaction module
"""

from typing import Union
from mindspore.nn import Cell

from interaction import Interaction, _INTERACTION_BY_KEY
from schnet import SchNetInteraction
from physnet import PhysNetModule
from niu import NeuralInteractionUnit

__all__ = [
    'Interaction',
    'SchNetInteraction',
    'PhysNetModule',
    'NeuralInteractionUnit',
    'get_interaction',
]


_INTERACTION_BY_NAME = {inter.__name__: inter for inter in _INTERACTION_BY_KEY.values()}


def get_interaction(cls_name: Union[Interaction, str, dict],
                    dim_node_rep: int,
                    dim_edge_rep: int,
                    dim_node_emb: int = None,
                    dim_edge_emb: int = None,
                    activation: Cell = None,
                    **kwargs) -> Interaction:
    """get molecular model"""
    if cls_name is None or isinstance(cls_name, Interaction):
        return cls_name

    if isinstance(cls_name, dict):
        return get_interaction(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _INTERACTION_BY_KEY.keys():
            return _INTERACTION_BY_KEY[cls_name.lower()](
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                dim_node_emb=dim_node_emb,
                dim_edge_emb=dim_edge_emb,
                activation=activation,
                **kwargs
            )
        if cls_name in _INTERACTION_BY_NAME.keys():
            return _INTERACTION_BY_NAME[cls_name](
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                dim_node_emb=dim_node_emb,
                dim_edge_emb=dim_edge_emb,
                activation=activation,
                **kwargs
            )
        raise ValueError("The Interaction corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported MolecularGNN type '{}'.".format(type(cls_name)))
