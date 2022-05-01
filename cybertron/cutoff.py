# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
# 
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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

import math
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

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
    def __init__(self, cutoff: float):

        super().__init__()
        self.reg_key = 'none'
        self.name = 'cutoff'

        self.cutoff = cutoff

        self.inv_cutoff = msnp.reciprocal(self.cutoff)

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.inv_cutoff = msnp.reciprocal(self.cutoff)
        return self

@_cutoff_register('cosine')
class CosineCutoff(Cutoff):
    r"""Class of Behler cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): cutoff radius.

    """

    def __init__(self, cutoff: float):
        super().__init__(cutoff=cutoff)

        self.reg_key = 'cosine'
        self.name = 'cosine cutoff'

        self.pi = Tensor(math.pi,ms.float32)
        self.cos = P.Cos()
        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbour_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        
        cuts = 0.5 * (self.cos(distances * self.pi * self.inv_cutoff) + 1.0)

        mask = distances < self.cutoff
        if neighbour_mask is not None:
            mask = self.logical_and(mask, neighbour_mask)

        # Remove contributions beyond the cutoff radius
        cutoffs = cuts * mask

        return cutoffs, mask

@_cutoff_register('mollifier')
class MollifierCutoff(Cutoff):
    r"""Class for mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float, optional): Cutoff radius.
        eps (float, optional): offset added to distances for numerical stability.

    """

    def __init__(
        self,
        cutoff: float,
        eps: float=1e-8,
    ):
        super().__init__(cutoff=cutoff)

        self.reg_key = 'mollifier'
        self.name = "Mollifier cutoff"
        self.eps = eps

        self.exp = P.Exp()
        self.logical_and = P.LogicalAnd()

    def set_eps(self,eps):
        self.eps = eps
        return self

    def construct(self, distances, neighbour_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """

        exponent = 1.0 - msnp.reciprocal(1.0 - F.square(distances * self.inv_cutoff))
        cutoffs = self.exp(exponent)
        
        mask = (distances + self.eps) < self.cutoff
        if neighbour_mask is not None:
            mask = self.logical_and(mask,neighbour_mask)

        cutoffs = cutoffs * mask

        return cutoffs, mask

@_cutoff_register('hard')
class HardCutoff(Cutoff):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(self, cutoff: float):
        super().__init__(cutoff=cutoff)

        self.reg_key = 'hard'
        self.name = "Hard cutoff"

        self.logical_and = P.LogicalAnd()

    def construct(self, distances, neighbour_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor): values of interatomic distances.

        Returns:
            mindspore.Tensor: values of cutoff function.

        """

        mask = distances < self.cutoff

        if neighbour_mask is not None:
            self.logical_and(mask,neighbour_mask)

        return F.cast(mask,distances.dtype), mask

@_cutoff_register('smooth')
class SmoothCutoff(Cutoff):
    r"""Class of smooth cutoff by Ebert, D. S. et al:
        [ref] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003

    ..  math::
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
        d_max (float, optional): the maximum distance (cutoff radius).
        d_min (float, optional): the minimum distance

    """
    def __init__(self, cutoff: float):
        super().__init__(cutoff=cutoff)

        self.reg_key = 'smooth'
        self.name = 'Smooth Cutoff'

        self.pow = P.Pow()
        self.logical_and = P.LogicalAnd()
        
    def construct(self, distance, neighbour_mask=None):
        """Compute cutoff.

        Args:
            distances (mindspore.Tensor or float): values of interatomic distances.

        Returns:
            mindspore.Tensor or float: values of cutoff function.

        """
        dd = distance * self.inv_cutoff
        cuts =  -  6. * self.pow(dd,5) \
                + 15. * self.pow(dd,4) \
                - 10. * self.pow(dd,3)

        cutoffs = 1 + cuts
        mask_upper = distance > 0
        mask_lower = distance < self.cutoff
        
        if neighbour_mask is not None:
            mask_lower = self.logical_and(mask_lower,neighbour_mask)
        
        cutoffs = msnp.where(mask_upper, cutoffs, 1)
        cutoffs = msnp.where(mask_lower, cutoffs, 0)
        
        return cutoffs, mask_lower

@_cutoff_register('gaussian')
class GaussianCutoff(Cutoff):
    r"""Class of hard cutoff.

    .. math::
       f(r) = \begin{cases}
        1 & r \leqslant r_\text{cutoff} \\
        0 & r > r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.

    """

    def __init__(
        self,
        cutoff: float,
        sigma: float=None,
    ):
        super().__init__(cutoff=cutoff)

        self.reg_key = 'gaussian'
        self.name = 'Gaussian Cutoff'

        self.sigma = sigma
        if self.sigma is None:
            self.sigma = cutoff
        self.inv_sigma2 = msnp.reciprocal(self.sigma * self.sigma)

        self.exp = P.Exp()
        self.logical_and = P.LogicalAnd()

    def set_sigma(self,sigma):
        self.sigma = sigma
        self.inv_sigma2 = msnp.reciprocal(self.sigma * self.sigma)
        return self

    def construct(self, distance, neighbour_mask=None):
        dd = distance - self.cutoff
        dd2 = dd * dd

        gauss = self.exp(-0.5 * dd2 * self.inv_sigma2)

        cuts = 1. - gauss
        mask = distance < self.cutoff
        if neighbour_mask is not None:
            mask = self.logical_and(mask,neighbour_mask)
        
        cuts = cuts * mask

        return cuts, mask

_CUTOFF_BY_NAME = {cut.__name__:cut for cut in _CUTOFF_BY_KEY.values()}

def get_cutoff(cutoff_fn:str,cutoff_dis:float) -> Cutoff:
    if cutoff_fn is None:
        return None
    if isinstance(cutoff_fn,Cutoff):
        return cutoff_fn

    elif isinstance(cutoff_fn, str):
        if cutoff_fn.lower() == 'none':
            return None
        if cutoff_fn.lower() in _CUTOFF_BY_KEY.keys():
            return _CUTOFF_BY_KEY[cutoff_fn.lower()](cutoff=cutoff_dis)
        elif cutoff_fn in _CUTOFF_BY_NAME.keys():
            return _CUTOFF_BY_NAME[cutoff_fn](cutoff=cutoff_dis)
        else:
            raise ValueError("The Cutoff corresponding to '{}' was not found.".format(cutoff_fn))
    else:
        raise TypeError("Unsupported Cutoff type '{}'.".format(type(cutoff_fn)))