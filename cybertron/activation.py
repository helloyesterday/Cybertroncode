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

from multiprocessing.sharedctypes import Value
from mindspore import nn
from mindspore.nn import Cell
# from mindspore.nn.layer import .activation import _activation
from mindspore.nn.layer import activation
from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck

__all__ = [
    "ShiftedSoftplus",
    "Swish",
    "get_activation",
    "get_activation_key",
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
        self.ln2 = 0.6931471805599453

    def __str__(self):
        return 'ShiftedSoftplus<>'

    def construct(self,x):
        # return self.softplus(x) - self.ln2
        return self.log1p(self.exp(x)) - self.ln2

class Swish(Cell):
    r"""Compute swish\SILU\SiL function.

    .. math::
       y_i = x_i / (1 + e^{-beta * x_i})

    Args:
        x (mindspore.Tensor): input tensor.

    Returns:
        mindspore.Tensor: shifted soft-plus of input.

    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def __str__(self):
        return 'Swish<>'

    def construct(self,x):
        return x * self.sigmoid(x)

_ACTIVATIONS_BY_KEY = {
    'ssp': ShiftedSoftplus,
    'swish': Swish,
}
_ACTIVATIONS_BY_KEY.update(activation._activation)

_ACTIVATIONS_BY_NAME = {a.__name__:a for a in _ACTIVATIONS_BY_KEY.values()}

def get_activation(activation) -> Cell:
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

    if activation is None:
        return None

    if isinstance(activation,(Cell,Primitive,PrimitiveWithCheck,PrimitiveWithCheck)):
        return activation
    elif isinstance(activation,str):
        if activation.lower() == 'none':
            return None
        if activation.lower() in _ACTIVATIONS_BY_KEY.keys():
            return _ACTIVATIONS_BY_KEY[activation.lower()]()
        elif activation in _ACTIVATIONS_BY_NAME.keys():
            return _ACTIVATIONS_BY_NAME[activation]()
        else:
            raise ValueError("The activation corresponding to '{}' was not found.".format(activation))
    else:
        raise TypeError("Unsupported activation type: "+str(type(activation)))