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

from typing import Union, List
import numpy as np
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.numpy import count_nonzero
from mindspore.nn.loss.loss import LossBase

from mindsponge.function import get_arguments

from .wrapper import MoleculeWrapper
from ..cybertron import Cybertron


class MolWithEvalCell(MoleculeWrapper):
    r"""Basic cell to combine the network and the loss/evaluate function.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    """
    def __init__(self,
                 data_keys: List[str],
                 network: Cybertron,
                 loss_fn: Union[LossBase, List[LossBase]] = None,
                 loss_weights: List[Union[float, Tensor, ndarray]] = 1,
                 calc_force: bool = False,
                 energy_key: str = 'energy',
                 force_key: str = 'force',
                 weights_normalize: bool = False,
                 normed_evaldata: bool = False,
                 **kwargs
                 ):
        super().__init__(
            network=network,
            loss_fn=loss_fn,
            data_keys=data_keys,
            calc_force=calc_force,
            energy_key=energy_key,
            force_key=force_key,
            loss_weights=loss_weights,
            weights_normalize=weights_normalize,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self._normed_evaldata = normed_evaldata

        self._force_id = self.get_index(self.force_key, self.label_keys)

        if loss_fn is not None:
            self._loss_fn = self._check_loss(loss_fn)
            self._loss_weights = self._check_weights(loss_weights)
            self._normal_factor = self._calc_normal_factor(self._loss_weights)
            self._molecular_loss = self._set_molecular_loss()
            self._any_atomwise = any(self._molecular_loss)
            self._set_atomwise_loss()

        self.zero = np.array(0, np.int32)

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.coordinate_id]
        atom_type = inputs[self.atom_type_id]
        pbc_box = inputs[self.pbc_box_id]
        neighbours = inputs[self.neighbours_id]
        neighbour_mask = inputs[self.neighbour_mask_id]
        bonds = inputs[self.bonds_id]
        bond_mask = inputs[self.bond_mask_id]

        labels = [inputs[self.labels_id[i]] for i in range(self.num_labels)]

        if atom_type is None:
            atom_type = self.atom_type
        atom_mask = atom_type > 0
        num_atoms = count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        normed_labels = None
        if self._normed_evaldata:
            normed_labels = labels
            labels = [self.scaleshift[i](normed_labels[i], atom_type, num_atoms)
                      for i in range(self.num_readouts)]
        elif self._loss_fn is not None:
            if self._force_id == -1:
                normed_labels = [self.scaleshift[i].normalize(labels[i], atom_type, num_atoms)
                                 for i in range(self.num_readouts)]
            else:
                normed_labels = []
                for i in range(self.num_labels):
                    if i == self._force_id:
                        normed_labels.append(self.scaleshift[i].normalize_force(labels[i]))
                    else:
                        normed_labels.append(self.scaleshift[i].normalize(labels[i], atom_type, num_atoms))

        outputs = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        if self.num_readouts == 1:
            outputs = (outputs,)

        def _normalize_outputs(outputs, outputs_scaled):
            if outputs_scaled:
                normed_outputs = ()
                if self._loss_fn is not None:
                    for i in range(self.num_readouts):
                        normed_outputs += (self.scaleshift[i].normalize(outputs[i], atom_type, num_atoms),)
            else:
                normed_outputs = outputs
                outputs = ()
                for i in range(self.num_readouts):
                    outputs += (self.scaleshift[i](normed_outputs[i], atom_type, num_atoms),)

            return outputs, normed_outputs

        outputs, normed_outputs = _normalize_outputs(outputs, self.outputs_scaled)

        if self.calc_force:
            force = -1 * self.grad_op(self._network)(
                coordinate,
                atom_type,
                pbc_box,
                neighbours,
                neighbour_mask,
                bonds,
                bond_mask,
            )

            if self.outputs_scaled:
                normed_force = None
                if self._loss_fn is not None:
                    normed_force = self.scaleshift[-1].normalize_force(force)
            else:
                normed_force = force
                force = self.scaleshift[-1].scale_force(normed_force)

            if self.num_labels == 1:
                outputs = (force,)
                normed_outputs = (normed_force,)
            else:
                outputs += (force,)
                normed_outputs += (normed_force,)

        
        if self._loss_fn is None:
            loss = self.zero
        else:
            loss = 0
            for i in range(self.num_labels):
                if self._molecular_loss[i]:
                    loss_ = self._loss_fn[i](normed_outputs[i], normed_labels[i], num_atoms, atom_mask)
                else:
                    loss_ = self._loss_fn[i](normed_outputs[i], normed_labels[i])

                loss += loss_ * self._loss_weights[i]
            loss *= self._normal_factor

        return loss, outputs, labels, num_atoms
