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
Cell for training and evaluation
"""

from typing import Union, List
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell, CellList
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import LossBase

from mindsponge.function import get_ms_array

from .loss import MolecularLoss
from ..normalize import ScaleShift
from ..cybertron import Cybertron


class MoleculeWrapper(Cell):
    r"""Base cell to combine the network and the loss/evaluate function.

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
                 **kwargs
                 ):

        super().__init__(auto_prefix=False)
        self._kwargs = kwargs

        if len(set(data_keys)) != len(data_keys):
            raise ValueError(f'Duplicate elements exist in data_keys: {data_keys}')

        self._network = network

        self.atomwise_readout = []
        for i in range(self.num_readouts):
            self.atomwise_readout.append(self._network.readout[i].shift_by_atoms)

        self._input_args = (
            'coordinate',
            'atom_type',
            'pbc_box',
            'neighbours',
            'neighbour_mask',
            'bonds',
            'bond_mask',
        )

        self.data_keys = data_keys
        self.num_data = len(self.data_keys)

        self.input_keys = []
        self.label_keys = []
        self.inputs = []
        self.labels_id = []
        for i, key in enumerate(self.data_keys):
            if key in self._input_args:
                self.inputs.append(i)
                self.input_keys.append(key)
            else:
                self.labels_id.append(i)
                self.label_keys.append(key)

        self.num_inputs = len(self.input_keys)
        self.num_labels = len(self.label_keys)

        self.coordinate_id = self.get_index('coordinate', self.data_keys)
        self.atom_type_id = self.get_index('atom_type', self.data_keys)
        self.pbc_box_id = self.get_index('pbc_box', self.data_keys)
        self.neighbours_id = self.get_index('neighbours', self.data_keys)
        self.neighbour_mask_id = self.get_index('neighbour_mask', self.data_keys)
        self.bonds_id = self.get_index('bonds', self.data_keys)
        self.bond_mask_id = self.get_index('bond_mask', self.data_keys)

        self.calc_force = calc_force

        if not self.calc_force and self.num_labels != self.num_outputs:
            raise ValueError(f'The number of network outputs is {self.num_outputs}, '
                             f'but the number of labels is {self.num_labels}. ')

        self.energy_key = energy_key
        self.force_key = force_key

        self.train_energy = self.num_readouts > 0

        if self.calc_force:
            if self.force_key not in self.label_keys:
                raise ValueError(f'Cannot find the key "{self.force_key} in labels: {self.label_keys}.')
            if self.label_keys.index(self.force_key) != len(self.label_keys) - 1:
                raise ValueError(f'The force in the label should be listed in the last place, '
                                 f'but got: {self.label_keys}')

            if self.num_labels == self.num_outputs:
                self.train_energy = False
            elif self.num_labels == self.num_outputs + 1:
                if self.energy_key not in self.label_keys:
                    raise ValueError(f'Cannot find the key "{self.energy_key} in labels: {self.label_keys}.')
                if self.label_keys.index(self.energy_key) != 0:
                    raise ValueError(f'The energy in the label should be listed in the first place, '
                                     f'but got: {self.label_keys}')
            else:
                raise ValueError(f'The number of network outputs is {self.num_outputs}, '
                                f'but the number of labels is {self.num_labels}. ')

        self._loss_fn: List[MolecularLoss] = loss_fn
        self._loss_weights = loss_weights
        self._molecular_loss = 1
        self._any_atomwise = False
        self._weights_normalize = weights_normalize
        self._normal_factor = 1

        self.atom_type = None
        if (context.get_context("mode") == context.PYNATIVE_MODE and
                'atom_type' in self._network.__dict__['_tensor_list'].keys()) or \
                (context.get_context("mode") == context.GRAPH_MODE and
                 'atom_type' in self._network.__dict__.keys()):
            self.atom_type = self._network.atom_type

        self.grad_op = C.GradOperation()

    @property
    def num_outputs(self) -> int:
        return self._network.num_outputs
    
    @property
    def num_readouts(self) -> int:
        return self._network.num_readouts
    
    @property
    def backbone_network(self):
        return self._network
    
    @property
    def scaleshift(self) -> List[ScaleShift]:
        return self._network.scaleshift
    
    @staticmethod
    def get_index(arg: str, data_keys: List[str]) -> int:
        if arg in data_keys:
            return data_keys.index(arg)
        return -1
    
    def _check_loss(self, loss_fn_) -> List[MolecularLoss]:
        if isinstance(loss_fn_, LossBase):
            loss_fn_ = [loss_fn_]
        if isinstance(loss_fn_, list):
            if len(loss_fn_) == self.num_labels:
                return CellList(loss_fn_)
            if len(loss_fn_) == 1:
                return CellList(loss_fn_ * self.num_labels)
            raise ValueError(f'The number of labels is {self.num_labels} but '
                                f'the number of loss_fn is {len(loss_fn_)}')
        else:
            raise TypeError(f'The type of loss_fn must be LossBase or lit of LossBase, '
                            f'but got: {type(loss_fn_)}')

    def _check_weights(self, weights_):
        if not isinstance(weights_, (list, tuple)):
            weights_ = [weights_]
        if len(weights_) != self.num_labels and len(weights_) == 1:
            weights_ = weights_ * self.num_labels
        if len(weights_) == self.num_labels:
            return [get_ms_array(w, ms.float32) for w in weights_]
        raise ValueError(f'The number of labels is {self.num_labels} but '
                            f'the number of loss_fn is {len(weights_)}')
    
    def _calc_normal_factor(self, loss_weights):
        if self._weights_normalize and self.num_labels > 1:
            normal_factor = 0
            for w in loss_weights:
                normal_factor += w
            return msnp.reciprocal(normal_factor)
        return 1
    
    def _set_molecular_loss(self):
        molecular_loss = []
        for i in range(self.num_labels):
            if isinstance(self._loss_fn[i], MolecularLoss):
                molecular_loss.append(True)
            else:
                molecular_loss.append(False)
        return molecular_loss
    
    def _set_atomwise_loss(self):
        if self.train_energy:            
            for i in range(self.num_readouts):
                if self._molecular_loss[i]:
                    self._loss_fn[i].set_atomwise(self.atomwise_readout[i])
        else:
            num = self.num_labels - 1 if self.calc_force else self.num_labels
            for i in range(num):
                if self._molecular_loss[i]:
                    self._loss_fn[i].set_atomwise(self.atomwise_readout[i+1])

        if self.calc_force and self._molecular_loss[-1]:
            self._loss_fn[-1].set_atomwise(True)


class WithAdversarialLossCell(Cell):
    r"""Adversarial network.

    Args:

        network (Cell): Neural network.

        loss_fn (Cell): Loss function.

    """
    def __init__(self,
                 network: Cell,
                 loss_fn: Cell,
                 ):

        super().__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, pos_samples: Tensor, neg_samples: Tensor):
        """calculate the loss function of adversarial network

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        pos_pred = self._network(pos_samples)
        neg_pred = self._network(neg_samples)
        return self._loss_fn(pos_pred, neg_pred)

    @property
    def backbone_network(self):
        return self._network
