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


@_cutoff_register('mollifier')
class MollifierCutoff(Cutoff):
    r"""mollifier cutoff network.

    Math:
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): Cutoff distance.

    """
    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray],
                 length_unit: Union[str, Units] = None,
                 eps: float = 1e-8,
                 **kwargs
                 ):
        super().__init__(
            cutoff=cutoff,
            length_unit=length_unit,
            )
        self._kwargs = get_arguments(locals(), kwargs)

        self.eps = get_ms_array(get_length(eps, self.units), ms.float32)

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        """Compute cutoff.

        Args:
            distances (Tensor):         Tensor of shape (..., K). Data type is float.
            neighbour_mask (Tensor):    Tensor of shape (..., K). Data type is bool.

        Returns:
            cutoff (Tensor):    Tensor of shape (..., K). Data type is float.

        """

        exponent = 1.0 - msnp.reciprocal(1.0 - F.square(distances * self.inv_cutoff))
        cutoffs = F.exp(exponent)

        mask = (distances + self.eps) < self.cutoff
        if neighbour_mask is not None:
            mask = F.logical_and(mask, neighbour_mask)

        cutoffs = cutoffs * mask

        return cutoffs, mask
