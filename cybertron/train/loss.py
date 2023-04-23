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
Loss functions
"""

import mindspore as ms
from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P
from mindspore.ops import functional as F


__all__ = [
    'LossWithEnergyAndForces',
    'MAELoss',
    'MSELoss'
    'CrossEntropyLoss',
]

class LossWithEnergyAndForces(LossBase):
    r"""Loss function of the energy and force of molecule.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 100,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(reduction)

        self.force_dis = Tensor(force_dis, ms.float32)
        self.ratio_normlize = ratio_normlize

        self.ratio_energy = ratio_energy
        self.ratio_forces = ratio_forces

        self.norm = 1
        if self.ratio_normlize:
            self.norm = ratio_energy + ratio_forces

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        """calculate loss function"""
        return diff

    def construct(self,
                  pred_energy: Tensor,
                  label_energy: Tensor,
                  pred_forces: Tensor = None,
                  label_forces: Tensor = None,
                  num_atoms: Tensor = 1,
                  atom_mask: Tensor = None
                  ):
        """calculate loss function

        Args:
            pred_energy (Tensor):   Tensor with shape (B, E). Data type is float.
                                    Predicted energy.
            label_energy (Tensor):  Tensor with shape (B, E). Data type is float.
                                    Label energy.
            pred_forces (Tensor):   Tensor with shape (B, A, D). Data type is float.
                                    Predicted force.
            label_forces (Tensor):  Tensor with shape (B, A, D). Data type is float.
                                    Label energy.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
                                    Number of atoms in each molecule.
                                    Default: 1
            atom_mask (Tensor):     Tensor with shape (B, A). Data type is bool.
                                    Mask of atoms in each molecule.
                                    Default: None

        Symbols:
            B:  Batch size
            A:  Number of atoms
            D:  Dimension of position coordinate. Usually is 3.
            E:  Number of labels

        Returns:
            loss (Tensor):  Tensor with shape (B, 1). Data type is float.
                            Loss function.

        """

        if pred_forces is None:
            loss = self._calc_loss(pred_energy - label_energy)
            return self.get_loss(loss)

        eloss = 0
        if self.ratio_forces > 0:
            ediff = (pred_energy - label_energy) / num_atoms
            eloss = self._calc_loss(ediff)

        floss = 0
        if self.ratio_forces > 0:
            # (B,A,D)
            fdiff = (pred_forces - label_forces) * self.force_dis
            fdiff = self._calc_loss(fdiff)
            # (B,A)
            fdiff = self.reduce_sum(fdiff, -1)

            if atom_mask is None:
                floss = self.reduce_mean(fdiff, -1)
            else:
                fdiff = fdiff * atom_mask
                floss = self.reduce_sum(fdiff, -1)
                floss = floss / num_atoms

        y = (eloss * self.ratio_energy + floss * self.ratio_forces) / self.norm

        natoms = F.cast(num_atoms, pred_energy.dtype)
        weights = natoms / self.reduce_mean(natoms)

        return self.get_loss(y, weights)


class MAELoss(LossWithEnergyAndForces):
    r"""Mean-absolute-error-type Loss function for energy and force.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 0,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )

        self.abs = P.Abs()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return self.abs(diff)


class MSELoss(LossWithEnergyAndForces):
    r"""Mean-square-error-type Loss function for energy and force.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 0,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )

        self.square = P.Square()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return self.square(diff)


class CrossEntropyLoss(LossBase):
    r"""Cross entropy Loss function for positive and negative samples.

    Args:

        reduction (str):    Method to reduction the output Tensor. Default: 'mean'

        use_sigmoid (bool): Whether to use sigmoid function for output. Default: False

    """
    def __init__(self,
                 reduction: str = 'mean',
                 use_sigmoid: bool = False
                 ):

        super().__init__(reduction)

        self.sigmoid = None
        if use_sigmoid:
            self.sigmoid = P.Sigmoid()

        self.cross_entropy = P.BinaryCrossEntropy(reduction)

    def construct(self, pos_pred: Tensor, neg_pred: Tensor):
        """calculate cross entropy loss function

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        if self.sigmoid is not None:
            pos_pred = self.sigmoid(pos_pred)
            neg_pred = self.sigmoid(neg_pred)

        pos_loss = self.cross_entropy(pos_pred, F.ones_like(pos_pred))
        neg_loss = self.cross_entropy(neg_pred, F.zeros_like(neg_pred))

        return pos_loss + neg_loss
