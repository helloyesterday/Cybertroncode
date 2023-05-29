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
from mindspore import Tensor
from mindspore.ops import functional as F

from mindsponge.function import get_ms_array, get_arguments
from mindsponge.function import Units, Length, get_length

from .cutoff import Cutoff, _cutoff_register


@_cutoff_register('gaussian')
class GaussianCutoff(Cutoff):
    r"""Gaussian-type cutoff network.

    Args:
        cutoff (float): Cutoff distance.

    """

    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray] = None,
                 sigma: Union[Length, float, Tensor, ndarray] = None,
                 **kwargs
                 ):
        super().__init__(cutoff=cutoff)
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'gaussian'
        self.name = 'Gaussian Cutoff'

        self.sigma = get_ms_array(sigma, ms.float32)

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
        
        sigma = self.sigma
        if self.sigma is None:
            sigma = cutoff
        
        dis = distance - cutoff
        dis2 = dis * dis
        decay = 1. - F.exp(-0.5 * dis2 * msnp.reciprocal(F.square(sigma)))

        if mask is None:
            mask = distance < cutoff
        else:
            mask = F.logical_and(distance < cutoff, mask)

        decay *= mask

        return decay, mask
