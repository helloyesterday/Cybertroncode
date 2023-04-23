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
from mindspore.ops import functional as F

from mindsponge.function import get_arguments
from mindsponge.function import Units, Length

from .cutoff import Cutoff, _cutoff_register


@_cutoff_register('hard')
class HardCutoff(Cutoff):
    r"""Hard cutoff network.

    Math:
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): Cutoff distance.

    """

    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray],
                 length_unit: Union[str, Units] = None,
                 **kwargs
                 ):
        super().__init__(
            cutoff=cutoff,
            length_unit=length_unit,
            )
        self._kwargs = get_arguments(locals(), kwargs)

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        """Compute cutoff.

        Args:
            distances (Tensor):         Tensor of shape (..., K). Data type is float.
            neighbour_mask (Tensor):    Tensor of shape (..., K). Data type is bool.

        Returns:
            cutoff (Tensor):    Tensor of shape (..., K). Data type is float.

        """

        mask = distances < self.cutoff

        if neighbour_mask is not None:
            F.logical_and(mask, neighbour_mask)

        return F.cast(mask, distances.dtype), mask
