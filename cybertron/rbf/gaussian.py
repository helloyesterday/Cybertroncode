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
Radical basis functions (RBF)
"""

from typing import Union
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from mindsponge.function import get_integer, get_ms_array, get_arguments
from mindsponge.function import Length, get_length

from .rbf import RadicalBasisFunctions, _rbf_register


@_rbf_register('gaussian')
class GaussianBasis(RadicalBasisFunctions):
    r"""Gaussian type RBF.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0 nm

        sigma (float):          Simga. Default: 0.03 nm

        delta (float):          Space interval. Default: 0.016 nm

        num_basis (int):        Number of basis functions. Defatul: None

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

    """

    def __init__(self,
                 r_max: Union[Length, float, Tensor, ndarray] = Length(1, 'nm'),
                 r_min: Union[Length, float, Tensor, ndarray] = 0,
                 sigma: Union[float, Tensor, ndarray] = Length(0.03, 'nm'),
                 delta: Union[float, Tensor, ndarray] = Length(0.016, 'nm'),
                 num_basis: int = None,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 **kwargs,
                 ):

        super().__init__(
            r_max=r_max,
            r_min=r_min,
            num_basis=num_basis,
            clip_distance=clip_distance,
            length_unit=length_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if num_basis is None and delta is None:
            raise TypeError('"num_basis" and "delta" cannot both be "None".')
        if num_basis is not None and num_basis <= 0:
            raise ValueError('"num_basis" must be larger than 0.')
        if delta is not None and delta <= 0:
            raise ValueError('"delta" must be larger than 0.')

        self.delta = get_ms_array(get_length(delta, self.units), ms.float32)
        self.sigma = get_ms_array(get_length(sigma, self.units), ms.float32)
        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))

        if self.delta is None:
            self.offsets = msnp.linspace(
                self.r_min, self.r_max, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(self.r_range/(self.num_basis-1), ms.float32)
        else:
            if self.num_basis is None:
                num_basis = msnp.ceil(self.r_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = self.r_min + \
                msnp.arange(0, self.num_basis) * self.delta

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
        print(ret+gap+' Sigma for Gaussian: ' +
              str(self.sigma)+' '+self.units.length_unit)
        print(ret+gap+' Interval for Gaussian: ' +
              str(self.delta)+' '+self.units.length_unit)
        print(ret+gap+' Number of basis functions: ' + str(self.num_basis))
        if self.clip_distance:
            print(ret+gap+' Clip the range of distance to (r_min,r_max).')
        return self

    def construct(self, distance: Tensor) -> Tensor:
        """Compute gaussian type RBF.

        Args:
            distance (Tensor): Tensor of shape `(...)`. Data type is float.

        Returns:
            rbf (Tensor): Tensor of shape `(..., K)`. Data type is float.

        """
        if self.clip_distance:
            distance = C.clip_by_value(distance, self.r_min, self.r_max)

        # (..., 1) <- (..., N)
        ex_dis = F.expand_dims(distance, -1)
        # (..., K) = (..., 1) - (K,)
        diff = ex_dis - self.offsets
        # (..., K)
        rbf = F.exp(self.coeff * F.square(diff))

        return rbf

    def __str__(self):
        return 'GaussianBasis<>'
