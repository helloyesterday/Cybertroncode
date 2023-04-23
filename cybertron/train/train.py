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
Cell for training
"""

from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from .cell import WithCell
from ..cybertron import Cybertron


__all__ = [
    "WithForceLossCell",
    "WithLabelLossCell",
]


class WithForceLossCell(WithCell):
    r"""Cell to combine the network and the loss function with force.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCNnBbFE'
        )

        #pylint: disable=invalid-name

        self.F = self.datatypes.find('F')  # force
        if self.F < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceLossCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.R]
        atom_type = inputs[self.Z]
        pbc_box = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        energy = inputs[self.E]
        out = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        forces = inputs[self.F]
        fout = -1 * self.grad_op(self._network)(
            coordinate,
            atom_type,
            pbc_box,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        if atom_type is None:
            atom_type = self.atom_type

        num_atoms = F.cast(atom_type > 0, out.dtype)
        num_atoms = self.keep_sum(num_atoms, -1)

        if atom_type is None:
            return self._loss_fn(out, energy, fout, forces)
        atom_mask = atom_type > 0
        return self._loss_fn(out, energy, fout, forces, num_atoms, atom_mask)

    @property
    def backbone_network(self):
        return self._network


class WithLabelLossCell(WithCell):
    r"""Cell to combine the network and the loss function with label.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCDNnBbE'
        )

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.R]
        atom_type = inputs[self.Z]
        pbc_box = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        out = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        label = inputs[self.E]

        if atom_type is None:
            atom_type = self.atom_type

        num_atoms = F.cast(atom_type > 0, out.dtype)
        num_atoms = self.keep_sum(num_atoms, -1)

        return self._loss_fn(out, label)
