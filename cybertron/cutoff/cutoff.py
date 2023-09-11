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

from typing import Union, Tuple
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor

from sponge.function import get_ms_array
from sponge.function import Units, GLOBAL_UNITS, Length, get_length


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
                 cutoff: Union[float, Tensor, ndarray] = None,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = kwargs

        self.cutoff = get_ms_array(cutoff, ms.float32)

    def set_cutoff(self, cutoff: Union[Length, float, Tensor, ndarray],
                   unit: Union[str, Units] = None):
        """set cutoff distance"""
        self.cutoff = get_ms_array(get_length(cutoff, unit), ms.float32)
        return self

    def construct(self,
                  distance: Tensor,
                  mask: Tensor = None,
                  cutoff: Tensor = None
                  ) -> Tuple[Tensor, Tensor]:
        """Compute cutoff.

        Args:
            distance (Tensor): Tensor of shape (..., K). Data type is float.
            mask (Tensor): Tensor of shape (..., K). Data type is bool.
            cutoff (Tensor): Tensor of shape (), (1,) or (..., K). Data type is float.

        Returns:
            decay (Tensor): Tensor of shape (..., K). Data type is float.
            mask (Tensor): Tensor of shape (..., K). Data type is bool.

        """
        raise NotImplementedError
