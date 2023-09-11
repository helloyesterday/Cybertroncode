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
from numpy import ndarray
from mindspore import Tensor
from mindspore.nn import Cell

from sponge.function import Units

from .readout import Readout, _READOUT_BY_KEY
from .node import NodeReadout, AtomwiseReadout, GraphReadout
from .gfn import GFNReadout

__all__ = [
    'Readout',
    'NodeReadout',
    'AtomwiseReadout',
    'GraphReadout',
    'GFNReadout',
    'get_readout',
]


_READOUT_BY_NAME = {out.__name__: out for out in _READOUT_BY_KEY.values()}


def get_readout(cls_name: Union[Readout, str],
                dim_node_rep: int = None,
                dim_edge_rep: int = None,
                activation: Union[Cell, str] = None,
                unit: Union[str, Units] = None,
                **kwargs,
                ) -> Readout:
    """get readout function

    Args:
        readout (str):          Name of readout function. Default: None
        model (MolecularGNN): Molecular model. Default: None
        dim_output (int):       Output dimension. Default: 1
        energy_unit (str):      Energy Unit. Default: None

    """
    if isinstance(cls_name, Readout):
        return cls_name
    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_readout(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _READOUT_BY_KEY.keys():
            return _READOUT_BY_KEY[cls_name.lower()](
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                activation=activation,
                unit=unit,
                **kwargs,
            )
        if cls_name in _READOUT_BY_NAME.keys():
            return _READOUT_BY_NAME[cls_name](
                dim_node_rep=dim_node_rep,
                dim_edge_rep=dim_edge_rep,
                activation=activation,
                unit=unit,
                **kwargs,
            )
        raise ValueError(
            "The Readout corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported Readout type '{}'.".format(type(cls_name)))
