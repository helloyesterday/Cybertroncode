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

import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.ops import functional as F

from mindsponge.function import get_arguments
from mindsponge.function import Units, Length

from .cutoff import Cutoff, _cutoff_register


@_cutoff_register('smooth')
class SmoothCutoff(Cutoff):
    r"""Smooth cutoff network.

    Reference:
        Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003

    Math:
        r_min < r < r_max:
        f(r) = 1.0 -  6 * ( r / r_cutoff ) ^ 5
                   + 15 * ( r / r_cutoff ) ^ 4
                   - 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 0
        r <= r_min: f(r) = 1

        reverse:
        r_min < r < r_max:
        f(r) =     6 * ( r / r_cutoff ) ^ 5
                - 15 * ( r / r_cutoff ) ^ 4
                + 10 * ( r / r_cutoff ) ^ 3
        r >= r_max: f(r) = 1
        r <= r_min: f(r) = 0

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
        dd = distances * self.inv_cutoff
        cuts = -  6. * F.pow(dd, 5) + 15. * F.pow(dd, 4) - 10. * F.pow(dd, 3)

        cutoffs = 1 + cuts
        mask_upper = distances > 0
        mask_lower = distances < self.cutoff

        if neighbour_mask is not None:
            mask_lower = F.logical_and(mask_lower, neighbour_mask)

        cutoffs = msnp.where(mask_upper, cutoffs, 1)
        cutoffs = msnp.where(mask_lower, cutoffs, 0)

        return cutoffs, mask_lower
