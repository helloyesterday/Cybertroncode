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
    'Error',
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
        label: ndarray = self._convert_data(labels)
        diff = label.reshape(predicts.shape) - predicts
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    r"""Metric to calcaulte the error.

    Args:

        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: False

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    """
    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = kwargs

        if not isinstance(index, int):
            raise TypeError(f'The type of index should be int but got: {type(index)}')

        self._index = get_integer(index)

        self._reduction = reduction
        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError(f'The type of "reduction" must be str, but got: {type(reduction)}')
            if reduction.lower() not in ('mean', 'sum', 'none'):
                raise ValueError(f"For '{self.__class__.__name__}', the 'reduction' must be in "
                                 f" ['mean', 'sum', 'none'], but got {reduction}.")
            self._reduction = reduction.lower()
            if self._reduction == 'none':
                self._reduction = None

        self._aggregate = aggregate
        if reduction is not None:
            if not isinstance(aggregate, str):
                raise TypeError(f'The type of "aggregate" must be str, but got: {type(aggregate)}')
            if aggregate.lower() not in ('mean', 'sum', 'none'):
                raise ValueError(f"For '{self.__class__.__name__}', the 'reduction' must be in "
                                 f" ['mean', 'sum', 'none'], but got {aggregate}.")
            self._aggregate = aggregate.lower()
            if self._aggregate == 'none':
                self._aggregate = None

        self._by_atoms = per_atom

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
               atom_mask: Tensor,
               ):
        """update metric"""
        #pylint: disable=unused-argument

        # (B, ...)
        predict = self._convert_data(predicts[self._index])
        # (B, ...)
        label = self._convert_data(labels[self._index])

        error: ndarray = self._calc_error(predict, label)
        batch_size = error.shape[0]

        if len(error.shape) > 2 and self._aggregate is not None:
            axis = tuple(range(2, len(error.shape)))
            # (B, A) <- (B, A, ...)
            if self._aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        num_atoms = 1
        total_num = batch_size
        if atom_mask is not None:
            atom_mask = self._convert_data(atom_mask)
            # (B, 1) <- (B, A) OR (1, 1) <- (1, A)
            num_atoms = np.count_nonzero(atom_mask, -1, keepdims=True)
            total_num = np.sum(num_atoms)
            if num_atoms.shape[0] == 1:
                total_num *= batch_size

        atomic = False
        if atom_mask is not None and error.shape[1] == atom_mask.shape[1]:
            atomic = True
            atom_mask_ = atom_mask
            if error.ndim != atom_mask.ndim:
                # (B, A, ...) <- (B, A)
                newshape = atom_mask.shape + (1,) * (error.ndim - atom_mask.ndim)
                atom_mask_ = np.reshape(atom_mask, newshape)
            # (B, A) * (B, A)
            error *= atom_mask_

        weight = batch_size
        if self._reduction is not None:
            error_shape1 = error.shape[1]
            # (B,) <- (B, ...)
            axis = tuple(range(1, len(error.shape)))
            error = np.sum(error, axis=axis)
            if self._reduction == 'mean':
                weight = batch_size * error_shape1
                if atomic or self._by_atoms:
                    weight = total_num

        self._error_sum += np.sum(error, axis=0)
        self._samples_num += weight

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

        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: False

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    """
    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.abs(label.reshape(predict.shape) - predict)


class MSE(Error):
    r"""Metric to calcaulte the mean square error.

    Args:

        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: False

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    """
    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.square(label.reshape(predict.shape) - predict)


class MNE(Error):
    r"""Metric to calcaulte the mean norm error.

    Args:

        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: False

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    """
    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        diff = label.reshape(predict.shape) - predict
        return np.linalg.norm(diff, axis=-1)


class RMSE(Error):
    r"""Metric to calcaulte the root mean square error.

    Args:

        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: False

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "sum".

    """
    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'sum',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
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
