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
Radical basis functions
"""

from typing import Union
from numpy import ndarray

from mindspore import Tensor
import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.function import Units, Length

from .rbf import RadicalBasisFunctions, _RBF_BY_KEY
from .gaussian import GaussianBasis
from .log_gaussian import LogGaussianBasis


__all__ = [
    'RadicalBasisFunctions',
    'GaussianBasis',
    'LogGaussianBasis',
    'get_rbf'
]


_RBF_BY_NAME = {rbf.__name__: rbf for rbf in _RBF_BY_KEY.values()}


def get_rbf(cls_name: Union[RadicalBasisFunctions, str, dict],
            r_max: Union[Length, float, Tensor, ndarray] = Length(1, 'nm'),
            num_basis: int = None,
            length_unit: Union[str, Units] = 'nm',
            **kwargs
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
