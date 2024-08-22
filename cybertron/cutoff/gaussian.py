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
from mindspore import Tensor
from mindspore.ops import functional as F

import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.function import get_ms_array, get_arguments
from sponge.function import Units, Length, get_length

from .cutoff import Cutoff, _cutoff_register


@_cutoff_register('gaussian')
class GaussianCutoff(Cutoff):
    r"""Gaussian-type cutoff network.

    Args:
        cutoff (float): Cutoff distance.

    """

    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray],
                 sigma: Union[Length, float, Tensor, ndarray] = None,
                 length_unit: Union[str, Units] = None,
                 **kwargs
                 ):
        super().__init__(
            cutoff=cutoff,
            length_unit=length_unit,
            )
        self._kwargs = get_arguments(locals(), kwargs)

        self.reg_key = 'gaussian'
        self.name = 'Gaussian Cutoff'

        self.sigma = get_ms_array(get_length(sigma, self.units), ms.float32)
        if self.sigma is None:
            self.sigma = self.cutoff
        self.inv_sigma2 = msnp.reciprocal(self.sigma * self.sigma)

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        dd = distances - self.cutoff
        dd2 = dd * dd

        gauss = F.exp(-0.5 * dd2 * self.inv_sigma2)

        cuts = 1. - gauss
        mask = distances < self.cutoff
        if neighbour_mask is not None:
            mask = F.logical_and(mask, neighbour_mask)

        cuts = cuts * mask

        return cuts, mask
