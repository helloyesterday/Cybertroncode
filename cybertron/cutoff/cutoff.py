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

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor

from mindsponge.function import get_ms_array
from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length


_CUTOFF_BY_KEY = dict()


def _cutoff_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _CUTOFF_BY_KEY:
            _CUTOFF_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _CUTOFF_BY_KEY:
                _CUTOFF_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Cutoff(nn.Cell):
    r"""Cutoff network.

    Args:
        cutoff (float): Cutoff distance.

    """
    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray],
                 length_unit: Union[str, Units] = None,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = kwargs

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.reg_key = 'none'
        self.name = 'cutoff'

        self.cutoff = get_ms_array(get_length(cutoff, self.units), ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)

    def set_cutoff(self, cutoff: Union[Length, float, Tensor, ndarray],
                   unit: Union[str, Units] = None):
        """set cutoff distance"""
        self.cutoff = get_ms_array(get_length(cutoff, unit), ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)
        return self

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        """Compute cutoff.

        Args:
            distances (Tensor):         Tensor of shape (..., K). Data type is float.
            neighbour_mask (Tensor):    Tensor of shape (..., K). Data type is bool.

        Returns:
            cutoff (Tensor):    Tensor of shape (..., K). Data type is float.

        """
        raise NotImplementedError
