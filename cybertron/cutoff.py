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

import math
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.function import get_ms_array, get_arguments
from mindsponge.function import Units, GLOBAL_UNITS, Length, get_length

__all__ = [
    "CosineCutoff",
    "MollifierCutoff",
    "HardCutoff",
    "SmoothCutoff",
    "GaussianCutoff",
    "get_cutoff",
]

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
                 cutoff: Union[Length, float, Tensor, ndarray],
                 length_unit: Union[str, Units] = None,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.reg_key = 'none'
        self.name = 'cutoff'

        self.cutoff = get_ms_array(get_length(cutoff, self.units), ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)

    def set_cutoff(self, cutoff: Union[Length, float, Tensor, ndarray],
                   unit: Union[str, Units] = None):
        """set cutoff distance"""
        self.cutoff = get_ms_array(get_length(cutoff, unit), ms.float32)
        self.inv_cutoff = msnp.reciprocal(self.cutoff)
        return self

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        """Compute cutoff.

        Args:
            distances (Tensor):         Tensor of shape (..., K). Data type is float.
            neighbour_mask (Tensor):    Tensor of shape (..., K). Data type is bool.

        Returns:
            cutoff (Tensor):    Tensor of shape (..., K). Data type is float.

        """
        raise NotImplementedError


@_cutoff_register('cosine')
class CosineCutoff(Cutoff):
    r"""Cutoff network.

    Math:
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
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
            **kwargs
            )
        self._kwargs = get_arguments(locals(), kwargs)

        self.pi = Tensor(math.pi, ms.float32)
        self.cos = P.Cos()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances: Tensor, neighbour_mask: Tensor = None):
        """Compute cutoff.

        Args:
            distances (Tensor):         Tensor of shape (..., K). Data type is float.
            neighbour_mask (Tensor):    Tensor of shape (..., K). Data type is bool.

        Returns:
            cutoff (Tensor):    Tensor of shape (..., K). Data type is float.

        """

        cuts = 0.5 * (self.cos(distances * self.pi * self.inv_cutoff) + 1.0)

        mask = distances < self.cutoff
        if neighbour_mask is not None:
            mask = self.logical_and(mask, neighbour_mask)

        # Remove contributions beyond the cutoff radius
        cutoffs = cuts * mask

        return cutoffs, mask


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
            **kwargs
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
            **kwargs
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
            **kwargs
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
            **kwargs
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


_CUTOFF_BY_NAME = {cut.__name__: cut for cut in _CUTOFF_BY_KEY.values()}


def get_cutoff(cls_name: Union[Cutoff, str, dict],
               cutoff: Union[Length, float, Tensor, ndarray],
               length_unit: Union[str, Units] = None,
               **kwargs
               ) -> Cutoff:
    """get cutoff network by name"""
    if cls_name is None:
        return None
    if isinstance(cls_name, Cutoff):
        return cls_name

    if isinstance(cls_name, dict):
        return get_cutoff(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _CUTOFF_BY_KEY.keys():
            return _CUTOFF_BY_KEY[cls_name.lower()](cutoff=cutoff,
                                                    length_unit=length_unit,
                                                    **kwargs)
        if cls_name in _CUTOFF_BY_NAME.keys():
            return _CUTOFF_BY_NAME[cls_name](cutoff=cutoff,
                                             length_unit=length_unit,
                                             **kwargs)
        raise ValueError(
            "The Cutoff corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported Cutoff type '{}'.".format(type(cls_name)))
