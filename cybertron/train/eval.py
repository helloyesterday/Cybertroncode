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
Cell for evaluation
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from .cell import WithCell
from .normalize import OutputScaleShift, DatasetNormalization
from ..cybertron import Cybertron


__all__ = [
    "WithEvalCell",
    "WithForceEvalCell",
    "WithLabelEvalCell",
]


class WithEvalCell(WithCell):
    r"""Basic cell to combine the network and the evaluate function.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

        fulltypes (str):            Full list of data types. Default: RZCDNnBbE'

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 fulltypes: str = 'RZCNnBbE'
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes=fulltypes
        )

        self.scale = scale
        self.shift = shift

        if atomwise_scaleshift is None:
            atomwise_scaleshift = self._network.atomwise_scaleshift
        else:
            atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.atomwise_scaleshift = atomwise_scaleshift

        self.scaleshift = None
        self.normalization = None
        self.scaleshift_eval = eval_data_is_normed
        self.normalize_eval = False
        self.type_ref = None
        if scale is not None or shift is not None:
            if scale is None:
                scale = 1
            if shift is None:
                shift = 0

            if type_ref is not None:
                self.type_ref = Tensor(type_ref, ms.float32)

            self.scaleshift = OutputScaleShift(
                scale=scale,
                shift=shift,
                type_ref=self.type_ref,
                atomwise_scaleshift=atomwise_scaleshift
            )

            if self._loss_fn is not None:
                self.normalization = DatasetNormalization(
                    scale=scale,
                    shift=shift,
                    type_ref=self.type_ref,
                    atomwise_scaleshift=atomwise_scaleshift
                )
                if not eval_data_is_normed:
                    self.normalize_eval = True

            self.scale = self.scaleshift.scale
            self.shift = self.scaleshift.shift

            scale = self.scale.asnumpy().reshape(-1)
            shift = self.shift.asnumpy().reshape(-1)
            atomwise_scaleshift = self.scaleshift.atomwise_scaleshift.asnumpy().reshape(-1)
            print('   with scaleshift for training ' +
                  ('and evaluate ' if eval_data_is_normed else ' ')+'dataset:')
            if atomwise_scaleshift.size == 1:
                print('   Scale: '+str(scale))
                print('   Shift: '+str(shift))
                print('   Scaleshift mode: ' +
                      ('atomwise' if atomwise_scaleshift else 'graph'))
            else:
                print('   {:>6s}. {:>16s}{:>16s}{:>12s}'.format(
                    'Output', 'Scale', 'Shift', 'Mode'))
                for i, m in enumerate(atomwise_scaleshift):
                    scale_ = scale if scale.size == 1 else scale[i]
                    shift_ = scale if shift.size == 1 else shift[i]
                    mode = 'Atomwise' if m else 'graph'
                    print('   {:<6s}{:>16.6e}{:>16.6e}{:>12s}'.format(
                        str(i)+': ', scale_, shift_, mode))
            if type_ref is not None:
                print('   with reference value for atom types:')
                info = '   Type '
                for i in range(self.type_ref.shape[-1]):
                    info += '{:>10s}'.format('Label'+str(i))
                print(info)
                for i, ref in enumerate(self.type_ref):
                    info = '   {:<7s} '.format(str(i)+':')
                    for r in ref:
                        info += '{:>10.2e}'.format(r.asnumpy())
                    print(info)

        self.add_cast_fp32 = add_cast_fp32


class WithLabelEvalCell(WithEvalCell):
    r"""Cell to combine the network and the evaluate function with label.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCNnBbE',
        )

    def construct(self, *inputs):
        """calculate evaluate data

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):      Tensor of shape (B, 1). Data type is float.
                                Loss function of evaluate data.
            output (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Predicted results of network.
            label (Tensor):     Tensor of shape (B, 1). Data type is float.
                                Label of evaluate data.
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms in each molecule.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.R]
        atom_type = inputs[self.Z]
        pbc_box = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        label = inputs[self.E]
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(ms.float32, label)
            output = F.cast(output, ms.float32)

        if atom_type is None:
            atom_type = self.atom_type

        num_atoms = F.cast(atom_type > 0, ms.int32)
        num_atoms = msnp.sum(atom_type > 0, -1, keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            if self.normalize_eval:
                normed_label = self.normalization(label, num_atoms, atom_type)
                loss = self._loss_fn(output, normed_label)
            else:
                loss = self._loss_fn(output, label)

        if self.scaleshift is not None:
            output = self.scaleshift(output, num_atoms, atom_type)
            if self.scaleshift_eval:
                label = self.scaleshift(label, num_atoms, atom_type)

        return loss, output, label, num_atoms


class WithForceEvalCell(WithEvalCell):
    r"""Cell to combine the network and the evaluate function with force.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

    """
    def __init__(self,
                 datatypes,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCNnBbFE',
        )
        #pylint: disable=invalid-name

        self.F = self.datatypes.find('F')  # force

        if self.F < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceEvalCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """calculate evaluate data

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):      Tensor of shape (B, 1). Data type is float.
                                Loss function of evaluate data.
            output (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Predicted results of network.
            label (Tensor):     Tensor of shape (B, 1). Data type is float.
                                Label of evaluate data.
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms in each molecule.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.R]
        atom_type = inputs[self.Z]
        pbc_box = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output_energy = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        output_forces = -1 * self.grad_op(self._network)(
            coordinate,
            atom_type,
            pbc_box,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        label_forces = inputs[self.F]
        label_energy = inputs[self.E]

        if self.add_cast_fp32:
            label_forces = F.mixed_precision_cast(ms.float32, label_forces)
            label_energy = F.mixed_precision_cast(ms.float32, label_energy)
            output_energy = F.cast(output_energy, ms.float32)

        if atom_type is None:
            atom_type = self.atom_type

        num_atoms = F.cast(atom_type > 0, ms.int32)
        num_atoms = msnp.sum(atom_type > 0, -1, keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            atom_mask = atom_type > 0
            if self.normalize_eval:
                normed_label_energy = self.normalization(
                    label_energy, num_atoms, atom_type)
                normed_label_forces = label_forces / self.scale
                loss = self._loss_fn(output_energy, normed_label_energy,
                                     output_forces, normed_label_forces, num_atoms, atom_mask)
            else:
                loss = self._loss_fn(
                    output_energy, label_energy, output_forces, label_forces, num_atoms, atom_mask)

        if self.scaleshift is not None:
            output_energy = self.scaleshift(
                output_energy, num_atoms, atom_type)
            output_forces = output_forces * self.scale
            if self.scaleshift_eval:
                label_energy = self.scaleshift(
                    label_energy, num_atoms, atom_type)
                label_forces = label_forces * self.scale

        return loss, output_energy, label_energy, output_forces, label_forces, num_atoms
