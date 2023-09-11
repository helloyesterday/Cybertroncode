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
                 cutoff: Union[Length, float, Tensor, ndarray] = None,
                 eps: float = 1e-8,
                 **kwargs
                 ):
        super().__init__(cutoff=cutoff)
        self._kwargs = get_arguments(locals(), kwargs)

        self.eps = get_ms_array(get_length(eps, self.units), ms.float32)

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

        exponent = 1.0 - msnp.reciprocal(1.0 - F.square(distance * msnp.reciprocal(cutoff)))
        decay = F.exp(exponent)
        
        if mask is None:
            mask = (distance + self.eps) < cutoff
        else:
            mask = F.logical_and((distance + self.eps) < cutoff, mask)

        decay *= mask

        return decay, mask
