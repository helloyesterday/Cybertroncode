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

from mindsponge.function import get_integer, get_ms_array, get_arguments
from mindsponge.function import keepdims_sum

from mindspore.nn.loss.loss import LossBase
from ..cybertron import Cybertron


__all__ = [
    "WithCell",
    "WithAdversarialLossCell",
]


class MoleculeWithLossCell(Cell):
    r"""Basic cell to combine the network and the loss/evaluate function.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

        fulltypes (str):        Full list of data types. Default: RZCDNnBbE'

    """
    def __init__(self,
                 network: Cybertron,
                 loss_fn: Union[LossBase, List[LossBase]],
                 data_keys: List[str],
                 force_keys: str = 'force',
                 loss_weights: List[float] = 1,
                 weights_normalize: bool = False,
                 **kwargs
                 ):
        super().__init__(auto_prefix=False)
        self._kwargs = get_arguments(locals(), kwargs)

        if len(set(data_keys)) != len(data_keys):
            raise ValueError(f'Duplicate elements exist in data_keys: {data_keys}')

        self._network = network

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
        self.labels = []
        for i, key in enumerate(self.data_keys):
            if key in self._input_args:
                self.inputs.append(i)
                self.input_keys.append(key)
            else:
                self.labels.append(i)
                self.label_keys.append(key)

        self.num_inputs = len(self.input_keys)
        self.num_labels = len(self.label_keys)

        self.coordinate = self.get_index('coordinate', self.data_keys)
        self.atom_type = self.get_index('atom_type', self.data_keys)
        self.pbc_box = self.get_index('pbc_box', self.data_keys)
        self.neighbours = self.get_index('neighbours', self.data_keys)
        self.neighbour_mask = self.get_index('neighbour_mask', self.data_keys)
        self.bonds = self.get_index('bonds', self.data_keys)
        self.bond_mask = self.get_index('bond_mask', self.data_keys)

        if self.num_labels < self.num_outputs:
            raise ValueError(f'The number of labels ({self.num_labels} cannot be less than '
                             f'the number of model outputs ({self.num_outputs}))')

        self.force_keys = force_keys

        self.calc_force = False
        self.force_in_label = None
        if self.num_labels > self.num_outputs:
            if self.num_outputs == 1 and self.num_labels == 2 and self.force_keys in self.label_keys:
                self.calc_force = True
                self.force_in_label = self.label_keys.index(self.force_keys)
            else:
                raise ValueError(f'The number of network outputs is {self.num_outputs}, '
                                 f'but the number of labels is {self.num_labels}: {self.label_keys}.')
            
        def _check_loss(loss_fn_):
            if self.num_labels == 1:
                if isinstance(loss_fn_, LossBase):
                    return loss_fn_
                if isinstance(loss_fn_, (list, tuple)):
                    if len(loss_fn_) == 1:
                        return loss_fn_[0]
                    raise ValueError(f'The number of labels is 1 but the number of loss_fn is {len(loss_fn_)}')
                raise TypeError(f'The type of loss_fn must be LossBase but got: {type(loss_fn_)}')
            else:
                if isinstance(loss_fn_, LossBase):
                    loss_fn_ = [loss_fn_]
                if isinstance(loss_fn_, list):
                    if len(loss_fn_) == self.num_labels:
                        return CellList(loss_fn_)
                    if len(loss_fn_) == 1:
                        return CellList(loss_fn_ * self.num_labels)
                    raise ValueError(f'The number of labels is {self.num_labels} but '
                                    f'the number of loss_fn is {len(loss_fn_)}')
                
        def _check_weights(weights_):
            if self.num_labels > 1:
                if not isinstance(weights_, (list, tuple)):
                    weights_ = [weights_]
                if len(weights_) != self.num_labels and len(weights_) == 1:
                    weights_ = weights_ * self.num_labels
                if len(weights_) == self.num_labels:
                    return [get_ms_array(w, ms.float32) for w in weights_]
                raise ValueError(f'The number of labels is {self.num_labels} but '
                                f'the number of loss_fn is {len(weights_)}')
            return weights_

        self._loss_fn = _check_loss(loss_fn)
        self.loss_weights= _check_weights(loss_weights)

        self.normal_factor = 1
        if weights_normalize and self.num_labels > 1:
            normal_factor = 0
            for w in self.loss_weights:
                normal_factor += w
            self.normal_factor = msnp.reciprocal(normal_factor)

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

    def get_index(self, arg: str, data_keys: List[str]) -> int:
        if arg in data_keys:
            return data_keys.index(arg)
        return -1

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        coordinate = inputs[self.coordinate]
        atom_type = inputs[self.atom_type]
        pbc_box = inputs[self.pbc_box]
        neighbours = inputs[self.neighbours]
        neighbour_mask = inputs[self.neighbour_mask]
        bonds = inputs[self.bonds]
        bond_mask = inputs[self.bond_mask]

        labels = [inputs[self.labels[i]] for i in range(self.num_labels)]

        outputs = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

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
            if self.force_in_label == 0:
                outputs = (force, outputs)
            else:
                outputs = (outputs, force)

        if atom_type is None:
            atom_type = self.atom_type

        num_atoms = F.cast(atom_type > 0, outputs.dtype)
        num_atoms = keepdims_sum(num_atoms, -1)

        atom_mask = atom_type > 0

        if self.num_labels == 1:
            return self._loss_fn(outputs, labels, num_atoms, atom_mask)
        else:
            loss = 0
            for i in range(self.num_labels):
                loss += self._loss_fn[i](outputs[i], labels[i], num_atoms, atom_mask) * self.loss_weights[i]
            return loss * self.normal_factor


class WithCell(Cell):
    r"""Basic cell to combine  the network and the loss/evaluate function.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

        fulltypes (str):        Full list of data types. Default: RZCDNnBbE'

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 fulltypes: str = 'RZCNnBbE',
                 ):

        super().__init__(auto_prefix=False)

        #pylint: disable=invalid-name

        self.fulltypes = fulltypes
        self.datatypes = datatypes

        if not isinstance(self.datatypes, str):
            raise TypeError('Type of "datatypes" must be str')

        for datatype in self.datatypes:
            if self.fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in self.fulltypes:
            num = self.datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype +
                                 '" in datatype "' + self.datatypes + '".')

        self.R = self.datatypes.find('R')  # coordinate
        self.Z = self.datatypes.find('Z')  # atom_type
        self.C = self.datatypes.find('C')  # pbcbox
        self.N = self.datatypes.find('N')  # neighbours
        self.n = self.datatypes.find('n')  # neighbour_mask
        self.B = self.datatypes.find('B')  # bonds
        self.b = self.datatypes.find('b')  # bond_mask
        self.E = self.datatypes.find('E')  # energy

        if self.E < 0:
            raise TypeError('The datatype "E" must be included!')

        self._network = network
        self._loss_fn = loss_fn

        self.atom_type = None
        if (context.get_context("mode") == context.PYNATIVE_MODE and
                'atom_type' in self._network.__dict__['_tensor_list'].keys()) or \
                (context.get_context("mode") == context.GRAPH_MODE and
                 'atom_type' in self._network.__dict__.keys()):
            self.atom_type = self._network.atom_type

        print(self.cls_name + ' with input type: ' + self.datatypes)


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
