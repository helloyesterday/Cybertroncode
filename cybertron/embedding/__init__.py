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

from mindsponge.function import Units, GLOBAL_UNITS

from .graph import GraphEmbedding, _EMBEDDING_BY_KEY
from .molecule import MolEmbedding
from .conformation import ConformationEmbedding

__all__ = [
    'GraphEmbedding',
    'MolEmbedding',
    'ConformationEmbedding',
    'get_embedding',
]


_EMBEDDING_BY_NAME = {filter.__name__: filter for filter in _EMBEDDING_BY_KEY.values()}


def get_embedding(cls_name: Union[GraphEmbedding, str],
                  dim_node: int,
                  dim_edge: int,
                  activation: Cell = None,
                  length_unit: Union[str, Units] = GLOBAL_UNITS.length_unit,
                  **kwargs,
                  ) -> GraphEmbedding:
    """get graph embedding by name"""

    if isinstance(cls_name, GraphEmbedding):
        return cls_name

    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_embedding(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _EMBEDDING_BY_KEY.keys():
            return _EMBEDDING_BY_KEY[cls_name.lower()](dim_node=dim_node,
                                                       dim_edge=dim_edge,
                                                       activation=activation,
                                                       length_unit=length_unit,
                                                       **kwargs
                                                       )
        if cls_name in _EMBEDDING_BY_NAME.keys():
            return _EMBEDDING_BY_NAME[cls_name](dim_node=dim_node,
                                                dim_edge=dim_edge,
                                                activation=activation,
                                                length_unit=length_unit,
                                                **kwargs
                                                )

        raise ValueError(
            "The filter corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported filter type '{}'.".format(type(cls_name)))
