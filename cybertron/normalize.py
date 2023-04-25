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
Modules for normalization
"""

from typing import Union, List
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.function import get_ms_array, get_arguments


__all__ = [
    'OutputScaleShift',
    'DatasetNormalization',
]


class ScaleShift(Cell):
    r"""A network to scale and shift the label of dataset or prediction.

    Args:

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        axis (int):                 Axis to summation the reference value of molecule. Default: -2

    Symbols:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        E:  Number of labels

    """

    def __init__(self,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 type_ref: Union[Tensor, ndarray] = None,
                 mode: str = 'atomwise',
                 axis: int = -2,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.scale = get_ms_array(scale, ms.float32)
        self.shift = get_ms_array(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = get_ms_array(type_ref, ms.float32)

        if mode.lower() in ['atomwise', 'a']:
            self.atomwise_readout = True
        elif mode.lower() in ['graph', 'set2set', 'g']:
            self.atomwise_readout = False
        else:
            self.atomwise_readout = None
            raise ValueError(f'Unknown mode: {mode}')

        self.axis = axis

    def construct(self, outputs: Tensor, num_atoms: Tensor, atom_type: Tensor = None):
        """Scale and shift output.

        Args:
            outputs (Tensor):       Tensor with shape (B, E). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):     Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B,E). Data type is float.

        """
        ref = 0
        if self.type_ref is not None:
            # (B,A,E)
            ref = F.gather(self.type_ref, atom_type, 0)
            # (B,E)
            ref = F.reduce_sum(ref, self.axis)

        # (B,E) + (B,E)
        outputs = outputs * self.scale + ref

        shift = self.shift
        if self.atomwise_readout:
            shift *= num_atoms

        # (B,E) + (B,1)
        return outputs + self.shift

class DatasetNormalization(Cell):
    r"""A network to normalize the label of dataset or prediction.

    Args:

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        axis (int):                 Axis to summation the reference value of molecule. Default: -2

    Symbols:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        E:  Number of labels

    """

    def __init__(self,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 type_ref: Union[Tensor, ndarray] = None,
                 mode: str = 'atomwise',
                 axis: int = -2,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.scale = get_ms_array(scale, ms.float32)
        self.shift = get_ms_array(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = get_ms_array(type_ref, ms.float32)

        if mode.lower() in ['atomwise', 'a']:
            self.atomwise_readout = True
        elif mode.lower() in ['graph', 'set2set', 'g']:
            self.atomwise_readout = False
        else:
            self.atomwise_readout = None
            raise ValueError(f'Unknown mode: {mode}')

        self.axis = axis

    def construct(self, label: Tensor, num_atoms: Tensor, atom_type: Tensor = None):
        """Normalize outputs.

        Args:
            outputs (Tensor):       Tensor with shape (B, E). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):    Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B,E). Data type is float.

        """
        ref = 0
        if self.type_ref is not None:
            ref = F.gather(self.type_ref, atom_type, 0)
            ref = F.reduce_sum(ref, self.axis)
        label -= ref

        shift = self.shift
        if self.atomwise_readout:
            shift *= num_atoms

        return (label - self.shift) / self.scale
