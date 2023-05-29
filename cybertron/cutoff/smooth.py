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
                 cutoff: Union[Length, float, Tensor, ndarray] = None,
                 **kwargs
                 ):
        super().__init__(cutoff=cutoff)
        self._kwargs = get_arguments(locals(), kwargs)

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

        if cutoff is None:
            cutoff = self.cutoff

        dis = distance * msnp.reciprocal(cutoff)
        decay = 1. - 6. * F.pow(dis, 5) + 15. * F.pow(dis, 4) - 10. * F.pow(dis, 3)

        mask_upper = distance > 0
        mask_lower = distance < cutoff
        if mask is not None:
            mask_lower = F.logical_and(mask_lower, mask)

        decay = msnp.where(mask_upper, decay, 1)
        decay = msnp.where(mask_lower, decay, 0)

        return decay, mask_lower
