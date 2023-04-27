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
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.function import get_ms_array, get_arguments
from mindsponge.function import Units, get_energy_unit


__all__ = [
    'OutputScaleShift',
    'DatasetNormalization',
]


class ScaleShift(Cell):
    r"""A network to scale and shift the label of dataset or prediction.

    Args:

        scale (float): Scale value. Default: 1

        shift (float): Shift value. Default: 0

        type_ref (Union[Tensor, ndarray]): Tensor of shape (T, E). Data type is float
            Reference values of label for each atom type. Default: None

        by_atoms (bool): Whether to do atomwise scale and shift. Default: None

        axis (int): Axis to summation the reference value of molecule. Default: -2

    Symbols:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        Y:  Number of labels

    """

    def __init__(self,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 type_ref: Union[Tensor, ndarray] = None,
                 shift_by_atoms: bool = True,
                 unit: str = None,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        try:
            self.output_unit = get_energy_unit(unit)
            self.units = Units(energy_unit=self.output_unit)
        except KeyError:
            self.output_unit = unit
            self.units = Units(energy_unit=None)

        scale = get_ms_array(scale, ms.float32)
        self.scale = Parameter(scale, name='scale', requires_grad=False)
        shift = get_ms_array(shift, ms.float32)
        self.shift = Parameter(shift, name='shift', requires_grad=False)

        type_ref = get_ms_array(type_ref, ms.float32)
        if type_ref is None:
            self.type_ref = Parameter(Tensor(0, ms.float32), name='type_ref', requires_grad=False)
        else:
            self.type_ref = Parameter(type_ref, name='type_ref', requires_grad=False)

        self.shift_by_atoms = shift_by_atoms

    def set_scaleshift(self,
                       scale: Union[float, Tensor, ndarray],
                       shift: Union[float, Tensor, ndarray],
                       type_ref: Union[Tensor, ndarray] = None):

        self.scale.set_data(get_ms_array(scale, ms.float32), True)
        self.shift.set_data(get_ms_array(shift, ms.float32), True)

        if type_ref is not None:
            self.type_ref.set_data(get_ms_array(type_ref, ms.float32), True)

        return self

    def convert_energy_from(self, unit) -> float:
        """returns a scale factor that converts the energy from a specified unit."""
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit) -> float:
        """returns a scale factor that converts the energy to a specified unit."""
        return self.units.convert_energy_to(unit)

    def set_unit(self, unit: str):
        """set output unit"""
        try:
            self.output_unit = get_energy_unit(unit)
            self.units.set_energy_unit(self.output_unit)
        except KeyError:
            self.output_unit = unit
            self.units.set_energy_unit(None)
        self._kwargs['unit'] = self.output_unit
        return self

    def normalize(self, label: Tensor, num_atoms: Tensor, atom_type: Tensor = None) -> Tensor:
        """Normalize outputs.

        Args:
            label (Tensor):       Tensor with shape (B, ...). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):    Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.

        """
        ref = 0
        if self.type_ref.ndim > 0:
            # (B, A, ...) <- (T, ...)
            ref = F.gather(self.type_ref, atom_type, 0)
            # (B, ...) <- (B, A, ...)
            ref = F.reduce_sum(ref, 1)

        # (B, ...) - (B, ...)
        label -= ref

        # (...)
        shift = self.shift
        if self.shift_by_atoms:
            if self.shift.ndim > 1:
                # (B, ...) <- (B, 1)
                num_atoms = F.reshape(num_atoms, (num_atoms.shape[0],) + (1,) * shift.ndim)
            # (B, ...) = (...) * (B, ...)
            shift *= num_atoms

        return (label - self.shift) / self.scale

    def construct(self, outputs: Tensor, num_atoms: Tensor, atom_type: Tensor = None) -> Tensor:
        """Scale and shift output.

        Args:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):     Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.

        """
        ref = 0
        if self.type_ref.ndim > 0:
            # (B, A, ...) <- (T, ...)
            ref = F.gather(self.type_ref, atom_type, 0)
            # (B, ...) <- (B, A, ...)
            ref = F.reduce_sum(ref, 1)

        # (B, ...) * (...) + (B, ...)
        outputs = outputs * self.scale + ref

        # (...)
        shift = self.shift
        if self.shift_by_atoms:
            if self.shift.ndim > 1:
                # (B, ...) <- (B, 1)
                num_atoms = F.reshape(num_atoms, (num_atoms.shape[0],) + (1,) * shift.ndim)
            # (B, ...) = (...) * (B, ...)
            shift *= num_atoms

        # (B, ...) + (B, ...)
        return outputs + self.shift
