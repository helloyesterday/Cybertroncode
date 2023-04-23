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
Readout functions
"""

from typing import Union, Tuple
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import functional as F

from mindsponge.function import Units, get_energy_unit
from mindsponge.function import get_integer, get_ms_array

from ..activation import get_activation


_READOUT_BY_KEY = dict()


def _readout_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _READOUT_BY_KEY:
            _READOUT_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _READOUT_BY_KEY:
                _READOUT_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Readout(Cell):
    r"""Readout function that merges and converts representation vectors into predicted properties.

    Args:

        dim_output (int): Dimension of outputs. Default: 1

        dim_node_rep (int): Dimension of node vectors. Default: None

        dim_edge_rep (int): Dimension of edge vectors. Default: None

        activation (Cell): Activation function, Default: None

        scale (float): Scale factor for outputs. Default: 1

        shift (float): Shift factor for outputs. Default: 0

        unit (str): Unit of output. Default: None

    Symbols:

        B: Batch size.

        A: Number of atoms.

        T: Number of atom types.

        Y: Output dimension.

    """
    def __init__(self,
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 activation: Cell = None,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 unit: str = None,
                 ndim: int = 1,
                 shape: Tuple[int] = (1,),
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        try:
            self.output_unit = get_energy_unit(unit)
            self.units = Units(energy_unit=self.output_unit)
        except KeyError:
            self.output_unit = unit
            self.units = Units(energy_unit=None)

        self.dim_node_rep = get_integer(dim_node_rep)
        self.dim_edge_rep = get_integer(dim_edge_rep)

        self._ndim = ndim
        self._shape = shape

        self.activation = None
        if activation is not None:
            self.activation = get_activation(activation)

        self.scale = Parameter(get_ms_array(scale, ms.float32), name='scale', requires_grad=False)
        self.shift = Parameter(get_ms_array(shift, ms.float32), name='shift', requires_grad=False)

    @property
    def ndim(self) -> int:
        """rank (ndim) of output Tensor (without batch size)"""
        return self._ndim

    @property
    def shape(self) -> Tuple[int]:
        """shape of output Tensor (without batch size)"""
        return self._shape

    @property
    def energy_unit(self) -> str:
        return self.units.energy_unit

    def set_dimension(self, dim_node_rep: int, dim_edge_rep: int):
        """check and set dimension of representation vectors"""
        if self.dim_node_rep is None:
            self.dim_node_rep = get_integer(dim_node_rep)
        elif self.dim_node_rep != dim_node_rep:
            raise ValueError(f'The `dim_node_rep` ({self.dim_node_rep}) of Readout cannot match '
                             f'the dimension of node representation vector ({dim_node_rep}).')

        if self.dim_edge_rep is None:
            self.dim_edge_rep = get_integer(dim_edge_rep)
        elif self.dim_edge_rep != dim_edge_rep:
            raise ValueError(f'The `dim_edge_rep` ({self.dim_edge_rep}) of Readout cannot match '
                             f'the dimension of edge representation vector ({dim_edge_rep}).')

        return self

    def convert_energy_from(self, unit) -> float:
        """returns a scale factor that converts the energy from a specified unit."""
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit) -> float:
        """returns a scale factor that converts the energy to a specified unit."""
        return self.units.convert_energy_to(unit)

    def set_unit(self, unit: str):
        """set output unit"""
        try:
            self.output_unit = get_energy_unit(unit)
            self.units.set_energy_unit(self.output_unit)
        except KeyError:
            self.output_unit = unit
            self.units.set_energy_unit(None)
        self._kwargs['unit'] = self.output_unit
        return self

    def set_scaleshift(self,
                       scale: Union[float, Tensor, ndarray] = 1,
                       shift: Union[float, Tensor, ndarray] = 0,
                       unit: str = None
                       ):
        """set scale and shift"""
        if unit is not None:
            self.set_unit(unit)

        scale: Tensor = get_ms_array(scale, ms.float32).reshape(-1)
        if scale.shape != self.shape and scale.size != 1:
            raise ValueError(f'The shape of "scale" ({scale.shape}) does not match'
                             f'the shape of output tensor ({self.shape})')
        if scale.shape == self.scale.shape:
            F.assign(self.scale, scale)
        else:
            self.scale = Parameter(scale, name='scale', requires_grad=False)

        shift: Tensor = get_ms_array(shift, ms.float32).reshape(-1)
        if shift.shape != self.shape and shift.size != 1:
            raise ValueError(f'The shape of "shift" ({shift.shape}) does not match'
                             f'the shape of output tensor ({self.shape})')
        if shift.shape == self.shift.shape:
            F.assign(self.shift, shift)
        else:
            self.shift = Parameter(shift, name='shift', requires_grad=False)

        return self

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f" Activation function: {self.activation}")
        print(ret+gap+f" Representation dimension: {self.dim_node_rep}")
        print(ret+gap+f" Shape of readout: {self.shape}")
        print(ret+gap+f" Rank (ndim) of readout: {self.ndim}")
        print(ret+gap+f" Scale: {self.scale.asnumpy()}")
        print(ret+gap+f" Shift: {self.shift.asnumpy()}")
        print(ret+gap+f" Output unit: {self.units.energy_unit_name}")
        print('-'*80)
        return self

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  neigh_dis: Tensor = None,
                  neigh_vec: Tensor = None,
                  neigh_list: Tensor = None,
                  neigh_mask: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs,
                  ):

        r"""Compute readout function.

        Args:
            node_rep (Tensor): Tensor of shape `(B, A, F)`. Data type is float.
                Atomic (node) representation vector.
            edge_rep (Tensor): Tensor of shape `(B, A, N, G)`. Data type is float.
                Edge representation vector.
            node_emb (Tensor): Tensor of shape `(B, A, E)`. Data type is float.
                Atomic (node) embedding vector.
            edge_emb (Tensor): Tensor of shape `(B, A, N, K)`. Data type is float.
                Edge embedding vector.
            atom_type (Tensor): Tensor of shape `(B, A)`. Data type is int.
                Index of atom types. Default: None
            atom_mask (Tensor): Tensor of shape `(B, A)`. Data type is bool
                Mask for atom types
            neigh_dis (Tensor): Tensor of shape `(B, A, N)`. Data type is float.
                Distances between central atom and its neighouring atoms.
            neigh_vec (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Vectors from central atom to its neighouring atoms.
            neigh_list (Tensor): Tensor of shape `(B, A, N)`. Data type is int.
                Indices of neighbouring atoms.
            neigh_mask (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Mask for neighbour list.
            bond_types (Tensor): Tensor of shape `(B, A, N)`. Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Mask for bonds

        Returns:
            output: (Tensor): Tensor of shape `(B, ...)`. Data type is float

        Symbols:
            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of node representation vector.
            G:  Feature dimension of edge representation vector.
            E:  Feature dimension of node embedding vector.
            K:  Feature dimension of edge embedding vector.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError
