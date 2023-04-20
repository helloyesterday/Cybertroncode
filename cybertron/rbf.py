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
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from mindsponge.data import set_hyper_parameter
from mindsponge.function import get_integer, get_ms_array, get_arguments
from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length

__all__ = [
    "GaussianBasis",
    "LogGaussianBasis",
    "get_rbf",
]

_RBF_BY_KEY = dict()


def _rbf_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _RBF_BY_KEY:
            _RBF_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _RBF_BY_KEY:
                _RBF_BY_KEY[alias] = cls
        return cls
    return alias_reg


class RadicalBasisFunctions(Cell):
    r"""Network of radical basis functions.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0 nm

        sigma (float):          Simga. Default: 0

        delta (float):          Space interval. Default: None

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: False

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

    """
    def __init__(self,
                 num_basis: int,
                 r_max: Union[Length, float, Tensor, ndarray],
                 r_min: Union[Length, float, Tensor, ndarray] = 0,
                 clip_distance: bool = False,
                 length_unit: Union[str, Units] ='nm',
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.num_basis = get_integer(num_basis)
        self.r_max = get_ms_array(get_length(r_max, self.units), ms.float32)
        self.r_min = get_ms_array(get_length(r_min, self.units), ms.float32)
        self.clip_distance = Tensor(clip_distance, ms.bool_)

        self.length_unit = self.units.length_unit

        if self.r_max <= self.r_min:
            raise ValueError('The argument "r_max" must be larger ' +
                             'than the argument "r_min" in RBF!')

        self.r_range = self.r_max - self.r_min

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        """print the information of RBF"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
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
        
        Symbol:
            K: Number of basis functions.

        """
        raise NotImplementedError


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
            **kwargs,
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


@_rbf_register('log_gaussian')
class LogGaussianBasis(RadicalBasisFunctions):
    r"""Log Gaussian type RBF.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0.04 nm

        sigma (float):          Simga. Default: 0.3

        delta (float):          Space interval. Default: 0.0512

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: True

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

        r_ref (Length):         Reference distance. Default: 1 nm

    """
    def __init__(self,
                 r_max: Union[Length, float, Tensor, ndarray] = Length(1, 'nm'),
                 r_min: Union[Length, float, Tensor, ndarray] = Length(0.04, 'nm'),
                 sigma: Union[Length, float, Tensor, ndarray] = 0.3,
                 delta: Union[Length, float, Tensor, ndarray] = 0.0512,
                 num_basis: int = None,
                 rescale: bool = True,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 r_ref: Length = Length(1, 'nm'),
                 **kwargs,
                 ):

        super().__init__(
            num_basis=num_basis,
            r_max=r_max,
            r_min=r_min,
            clip_distance=clip_distance,
            length_unit=length_unit,
            **kwargs,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if num_basis is None and delta is None:
            raise TypeError('"num_basis" and "delta" cannot both be "None".')
        if num_basis is not None and num_basis <= 0:
            raise ValueError('"num_basis" must be larger than 0.')
        if delta is not None and delta <= 0:
            raise ValueError('"delta" must be larger than 0.')
        if isinstance(delta, Length):
            raise TypeError(
                '"delta" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')
        
        self.delta = get_ms_array(delta, ms.float32)        
        
        if isinstance(sigma, Length):
            raise TypeError('"sigma" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')
        
        self.sigma = get_ms_array(sigma, ms.float32)
        self.rescale = rescale

        self.r_ref = get_ms_array(get_length(r_ref, self.units), ms.float32)

        log_rmin = msnp.log(self.r_min/self.r_ref, dtype=ms.float32)
        log_rmax = msnp.log(self.r_max/self.r_ref, dtype=ms.float32)
        log_range = log_rmax-log_rmin
        if self.delta is None:
            self.offsets = msnp.linspace(
                log_rmin, log_rmax, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(log_range/(self.num_basis-1), ms.float32)
        else:
            if self.num_basis is None:
                num_basis = msnp.ceil(log_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = msnp.log(self.r_min/self.r_ref) + \
                msnp.arange(0, self.num_basis) * self.delta

        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        self.inv_ref = msnp.reciprocal(self.r_ref)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
        print(ret+gap+' Reference distance: ' +
              str(self.r_ref)+' '+self.units.length_unit)
        print(ret+gap+' Log Gaussian begin: ' + str(self.offsets[0]))
        print(ret+gap+' Log Gaussian end: ' + str(self.offsets[-1]))
        print(ret+gap+' Interval for log Gaussian: '+str(self.delta))
        print(ret+gap+' Sigma for log gaussian: ' + str(self.sigma))
        print(ret+gap+' Number of basis functions: ' + str(self.num_basis))
        if self.clip_distance:
            print(ret+gap+' Clip the range of distance to (r_min,r_max).')
        if self.rescale:
            print(ret+gap+' Rescale the range of RBF to (-1,1).')
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

        # (...)
        log_r = F.log(distance * self.inv_ref)
        # (..., 1)
        log_r = F.expand_dims(log_r, -1)

        # (..., K) = (..., 1) - (K)
        log_diff = log_r - self.offsets
        rbf = F.exp(self.coeff*F.square(log_diff))

        if self.rescale:
            rbf = rbf * 2 - 1.0

        return rbf


_RBF_BY_NAME = {rbf.__name__: rbf for rbf in _RBF_BY_KEY.values()}


def get_rbf(cls_name: Union[RadicalBasisFunctions, str, dict],
            r_max: Union[Length, float, Tensor, ndarray] = Length(1, 'nm'),
            num_basis: int = None,
            length_unit: Union[str, Units] ='nm',
            **kwargs,
            ) -> RadicalBasisFunctions:
    """get RBF by name"""

    if isinstance(cls_name, RadicalBasisFunctions):
        return cls_name
    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_rbf(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _RBF_BY_KEY.keys():
            return _RBF_BY_KEY[cls_name.lower()](r_max=r_max,
                                            num_basis=num_basis,
                                            length_unit=length_unit,
                                            **kwargs)
        if cls_name in _RBF_BY_NAME.keys():
            return _RBF_BY_NAME[cls_name](r_max=r_max,
                                     num_basis=num_basis,
                                     length_unit=length_unit,
                                     **kwargs)

        raise ValueError(
            "The RBF corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported RBF type '{}'.".format(type(cls_name)))
