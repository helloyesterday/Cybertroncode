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

from typing import List, Tuple

import numpy as np
from numpy import ndarray
from mindspore import Tensor
from mindspore.nn.metrics import Metric

from mindsponge.function import get_integer, get_arguments


__all__ = [
    'MaxError',
    'Error'
    'MAE',
    'MSE',
    'MNE',
    'RMSE',
    'Loss',
]


class MaxError(Metric):
    r"""Metric to calcaulte the max error.

    Args:

        indexes (tuple):        Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool): Whether to summation the data of all atoms in molecule. Default: True

    """
    def __init__(self, index: int = 0, **kwargs):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.clear()
        self._indexes = get_integer(index)

    def clear(self):
        self._max_error = 0

    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               num_atoms: Tensor,
               *args):
        """update metric"""
        #pylint: disable=unused-argument

        predicts: ndarray = self._convert_data(predicts)
        labels: ndarray = self._convert_data(labels)
        diff = labels.reshape(predicts.shape) - predicts
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    r"""Metric to calcaulte the error.

    Args:

        indexes (tuple): Indexes for label and predicted data. Default: (1, 2)

        by_atoms (bool): Whether to average the data by the number of atoms. Default: True

        aggregate (str): The way to aggregate the data of each atom. Valid only for vector
            labels (e.g. force). Default: 'mean'

    """
    def __init__(self,
                 index: int = 0,
                 by_atoms: bool = False,
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = kwargs

        if not isinstance(index, int):
            raise TypeError(f'The type of index should be int but got: {type(index)}')
    
        self._index = get_integer(index)

        if aggregate.lower() not in ('mean', 'sum'):
            raise ValueError(f'aggregate method must be "mean" or "sum", but got: {aggregate}')

        self._aggregate = aggregate.lower()

        self._by_atoms = by_atoms

        self._error_sum = 0
        self._samples_num = 0

        self.clear()

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               num_atoms: Tensor,
               *args):
        """update metric"""
        #pylint: disable=unused-argument

        predict = self._convert_data(predicts[self._index])
        label = self._convert_data(labels[self._index])

        error: ndarray = self._calc_error(predict, label)
        if len(error.shape) > 2:
            axis = tuple(range(2, len(error.shape)))
            # (B, 1) <- (B, ...)
            if self._aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        tot = label.shape[0]
        if self._by_atoms:
            error /= self._convert_data(num_atoms)

        self._error_sum += np.sum(error)
        self._samples_num += tot

    def eval(self) -> float:
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        """calculate error"""
        raise NotImplementedError


class MAE(Error):
    r"""Metric to calcaulte the mean absolute error.

    Args:

        indexes (tuple): Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool): Whether to summation the data of all atoms in molecule. Default: True

        by_atoms (bool): Whether to averaged the data by the number of atoms in molecule.
            Default: True

        aggregate (str): The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 index: int = 0,
                 by_atoms: bool = False,
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            by_atoms=by_atoms,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.abs(label.reshape(predict.shape) - predict)


class MSE(Error):
    r"""Metric to calcaulte the mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 index: int = 0,
                 by_atoms: bool = False,
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            by_atoms=by_atoms,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(predict: ndarray, label: ndarray) -> ndarray:
        return np.square(label.reshape(predict.shape) - predict)


class MNE(Error):
    r"""Metric to calcaulte the mean norm error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 index: int = 0,
                 by_atoms: bool = False,
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            by_atoms=by_atoms,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        diff = label.reshape(predict.shape) - predict
        return np.linalg.norm(diff, axis=-1)


class RMSE(Error):
    r"""Metric to calcaulte the root mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 index: int = 0,
                 by_atoms: bool = False,
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            by_atoms=by_atoms,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.square(label.reshape(predict.shape) - predict)


class Loss(Metric):
    r"""Metric to calcaulte the loss function.

    Args:

        indexes (int):            Index for loss function. Default: 0

    """
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        self.clear()

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               num_atoms: Tensor,
               *args):
        #pylint: disable=unused-argument
        """update metric"""
        loss = self._convert_data(loss)

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
