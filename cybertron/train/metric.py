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
Metric functions
"""

import numpy as np
from numpy import ndarray
from mindspore.nn.metrics import Metric


__all__ = [
    'MaxError',
    'Error'
    'MAE',
    'MSE',
    'MNE',
    'RMSE',
    'MLoss',
]


class MaxError(Metric):
    r"""Metric to calcaulte the max error.

    Args:

        indexes (tuple):        Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool): Whether to summation the data of all atoms in molecule. Default: True

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True
                 ):

        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._max_error = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        diff = y.reshape(y_pred.shape) - y_pred
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    r"""Metric to calcaulte the error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all: bool = True,
                 by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__()
        self.clear()
        self._indexes = indexes
        self.read_num_atoms = False
        if len(self._indexes) > 2:
            self.read_num_atoms = True

        self.reduce_all = reduce_all

        if atom_aggregate.lower() not in ('mean', 'sum'):
            raise ValueError(
                'aggregate_by_atoms method must be "mean" or "sum"')
        self.atom_aggregate = atom_aggregate.lower()

        if reduce_all:
            self.axis = None
        else:
            self.axis = 0

        if by_atoms and not self.read_num_atoms:
            raise ValueError(
                'When to use averaged_by_atoms, the index of atom number must be set at "indexes".')

        self.by_atoms = by_atoms

        self._error_sum = 0
        self._samples_num = 0

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        """calculate error"""
        return y.reshape(y_pred.shape) - y_pred

    def update(self, *inputs):
        """update metric"""
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])

        error = self._calc_error(y, y_pred)
        if len(error.shape) > 2:
            axis = tuple(range(2, len(error.shape)))
            if self.atom_aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        tot = y.shape[0]
        if self.read_num_atoms:
            natoms = self._convert_data(inputs[self._indexes[2]])
            if self.by_atoms:
                error /= natoms
            elif self.reduce_all:
                tot = np.sum(natoms)
                if natoms.shape[0] != y.shape[0]:
                    tot *= y.shape[0]
        elif self.reduce_all:
            tot = error.size

        self._error_sum += np.sum(error, axis=self.axis)
        self._samples_num += tot

    def eval(self) -> float:
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num


class MAE(Error):
    r"""Metric to calcaulte the mean absolute error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all=reduce_all_dims,
            by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.abs(y.reshape(y_pred.shape) - y_pred)

class MSE(Error):
    r"""Metric to calcaulte the mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all=reduce_all_dims,
            by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.square(y.reshape(y_pred.shape) - y_pred)


class MNE(Error):
    r"""Metric to calcaulte the mean norm error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all=reduce_all_dims,
            by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        diff = y.reshape(y_pred.shape) - y_pred
        return np.linalg.norm(diff, axis=-1)

class RMSE(Error):
    r"""Metric to calcaulte the root mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all=reduce_all_dims,
            by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.square(y.reshape(y_pred.shape) - y_pred)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)


class MLoss(Metric):
    r"""Metric to calcaulte the loss function.

    Args:

        indexes (int):            Index for loss function. Default: 0

    """
    def __init__(self, index: int = 0):
        super().__init__()
        self.clear()
        self._index = index

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):
        """update metric"""
        loss = self._convert_data(inputs[self._index])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError(
                "Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num
