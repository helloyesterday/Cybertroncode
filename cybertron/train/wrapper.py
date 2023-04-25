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

from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from ..cybertron import Cybertron


__all__ = [
    "WithCell",
    "WithAdversarialLossCell",
]


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
