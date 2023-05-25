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
Cutoff functions
"""

from typing import Union
from numpy import ndarray

from mindspore import Tensor

from mindsponge.function import Units, Length

from .cutoff import Cutoff, _CUTOFF_BY_KEY
from .cosine import CosineCutoff
from .gaussian import GaussianCutoff
from .hard import HardCutoff
from .mollifier import MollifierCutoff
from .smooth import SmoothCutoff


__all__ = [
    'Cutoff',
    'CosineCutoff',
    'GaussianCutoff',
    'HardCutoff',
    'MollifierCutoff',
    'SmoothCutoff',
    'get_cutoff',
]


_CUTOFF_BY_NAME = {cut.__name__: cut for cut in _CUTOFF_BY_KEY.values()}


def get_cutoff(cls_name: Union[Cutoff, str, dict],
               cutoff: Union[Length, float, Tensor, ndarray],
               length_unit: Union[str, Units] = None,
               **kwargs
               ) -> Cutoff:
    """get cutoff network by name"""
    if cls_name is None:
        return None
    if isinstance(cls_name, Cutoff):
        return cls_name

    if isinstance(cls_name, dict):
        return get_cutoff(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _CUTOFF_BY_KEY.keys():
            return _CUTOFF_BY_KEY[cls_name.lower()](cutoff=cutoff,
                                                    length_unit=length_unit,
                                                    **kwargs)
        if cls_name in _CUTOFF_BY_NAME.keys():
            return _CUTOFF_BY_NAME[cls_name](cutoff=cutoff,
                                             length_unit=length_unit,
                                             **kwargs)
        raise ValueError(
            "The Cutoff corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported Cutoff type '{}'.".format(type(cls_name)))
