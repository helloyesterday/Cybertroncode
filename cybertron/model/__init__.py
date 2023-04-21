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

from typing import Union
from mindspore.nn import Cell
from mindsponge.function import Units

from model import MolecularGNN, _MODEL_BY_KEY
from schnet import SchNet
from physnet import PhysNet
from molct import MolCT

__all__ = [
    'SchNet',
    'PhysNet',
    'MolCT',
    'get_molecular_model',
]

_MODEL_BY_NAME = {model.__name__: model for model in _MODEL_BY_KEY.values()}


def get_molecular_model(cls_name: Union[MolecularGNN, str, dict],
                        dim_feature: int = 64,
                        n_interaction: int = 3,
                        coupled_interaction: bool = False,
                        use_graph_norm: bool = False,
                        activation: Cell = None,
                        length_unit: Union[str, Units] = None,
                        **kwargs) -> MolecularGNN:
    """get molecular model"""
    if cls_name is None or isinstance(cls_name, MolecularGNN):
        return cls_name

    if isinstance(cls_name, dict):
        return get_molecular_model(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _MODEL_BY_KEY.keys():
            return _MODEL_BY_KEY[cls_name.lower()](
                dim_feature=dim_feature,
                n_interaction=n_interaction,
                activation=activation,
                coupled_interaction=coupled_interaction,
                use_graph_norm=use_graph_norm,
                length_unit=length_unit,
                **kwargs
            )
        if cls_name in _MODEL_BY_NAME.keys():
            return _MODEL_BY_NAME[cls_name](
                dim_feature=dim_feature,
                n_interaction=n_interaction,
                activation=activation,
                coupled_interaction=coupled_interaction,
                use_graph_norm=use_graph_norm,
                length_unit=length_unit,
                **kwargs
            )
        raise ValueError("The MolecularGNN corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported MolecularGNN type '{}'.".format(type(cls_name)))
