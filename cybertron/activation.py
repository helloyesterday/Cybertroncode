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
Extra activation function
"""

import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore.nn.layer import activation
from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck

__all__ = [
    "ShiftedSoftplus",
    "get_activation",
]


class ShiftedSoftplus(Cell):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """

    def __init__(self):
        super().__init__()
        # self.softplus = P.Softplus()
        self.log1p = P.Log1p()
        self.exp = P.Exp()
        self.log2 = msnp.log(2.0)

    def __str__(self):
        return 'ShiftedSoftplus<>'

    def construct(self, x):
        # return self.softplus(x) - self.log2
        return self.log1p(self.exp(x)) - self.log2


_ACTIVATIONS_BY_KEY = {
    'ssp': ShiftedSoftplus,
}

#pylint: disable=protected-access
_ACTIVATIONS_BY_KEY.update(activation._activation)

_ACTIVATIONS_BY_NAME = {a.__name__: a for a in _ACTIVATIONS_BY_KEY.values()}


def get_activation(cls_name, **kwargs) -> Cell:
    """
    Gets the activation function.

    Args:
        obj (str): The obj of the activation function.

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sigmoid = nn.get_activation('sigmoid')
        >>> print(sigmoid)
        Sigmoid<>
    """

    if cls_name is None:
        return None

    if isinstance(cls_name, (Cell, Primitive, PrimitiveWithCheck,
                             PrimitiveWithInfer, PrimitiveWithCheck)):
        return cls_name
    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _ACTIVATIONS_BY_KEY.keys():
            return _ACTIVATIONS_BY_KEY[cls_name.lower()](**kwargs)
        if cls_name in _ACTIVATIONS_BY_NAME.keys():
            return _ACTIVATIONS_BY_NAME[cls_name](**kwargs)
        raise ValueError(
            "The activation corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported activation type: "+str(type(cls_name)))
