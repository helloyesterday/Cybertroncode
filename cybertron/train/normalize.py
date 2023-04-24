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

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F


__all__ = [
    'OutputScaleShift',
    'DatasetNormalization',
]


class OutputScaleShift(Cell):
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
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = None,
                 axis: int = -2,
                 ):

        super().__init__()

        self.scale = Tensor(scale, ms.float32)
        self.shift = Tensor(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref, ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True

        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(
                self.atomwise_scaleshift, (1, -1))

        self.axis = axis

    def construct(self, outputs: Tensor, num_atoms: Tensor, atom_type: Tensor = None):
        """Scale and shift output.

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
            # (B,A,E)
            ref = F.gather(self.type_ref, atom_type, 0)
            # (B,E)
            ref = F.reduce_sum(ref, self.axis)

        # (B,E) + (B,E)
        outputs = outputs * self.scale + ref
        if self.all_atomwsie:
            # (B,E) + (B,1)
            return outputs + self.shift * num_atoms
        if self.all_graph:
            # (B,E)
            return outputs + self.shift

        atomwise_output = outputs + self.shift * num_atoms
        graph_output = outputs + self.shift
        return msnp.where(self.atomwise_scaleshift, atomwise_output, graph_output)


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
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = None,
                 axis: int = -2,
                 ):

        super().__init__()

        self.scale = Tensor(scale, ms.float32)
        self.shift = Tensor(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref, ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True

        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(
                self.atomwise_scaleshift, (1, -1))

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
        if self.all_atomwsie:
            return (label - self.shift * num_atoms) / self.scale
        if self.all_graph:
            return (label - self.shift) / self.scale

        atomwise_norm = (label - self.shift * num_atoms) / self.scale
        graph_norm = (label - self.shift) / self.scale
        return msnp.where(self.atomwise_scaleshift, atomwise_norm, graph_norm)
