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

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, get_activation
from mindspore.numpy import count_nonzero
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.function import Units, get_energy_unit
from mindsponge.function import get_integer, get_ms_array, get_arguments

from .aggregator import Aggregator, get_aggregator
from .decoder import Decoder, get_decoder

__all__ = [
    "Readout",
    "NodeReadout",
]

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

        dim_node (int): Dimension of node vectors. Default: None

        dim_edge (int): Dimension of edge vectors. Default: None

        activation (Cell): Activation function, Default: None

        scale (float): Scale factor for outputs. Default: 1

        shift (float): Shift factor for outputs. Default: 0

        unit (str): Unit of output. Default: None

    Symbols:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """
    def __init__(self,
                 dim_output: int = 1,
                 dim_node: int = None,
                 dim_edge: int = None,
                 activation: Cell = None,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 unit: str = None,
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

        self.dim_output = get_integer(dim_output)
        self.dim_node = get_integer(dim_node)
        self.dim_edge = get_integer(dim_edge)

        self.activation = None
        if activation is not None:
            self.activation = get_activation(activation)

        self.scale = Parameter(get_ms_array(scale, ms.float32), name='scale', requires_grad=False)
        self.shift = Parameter(get_ms_array(shift, ms.float32), name='scale', requires_grad=False)

    @property
    def energy_unit(self) -> str:
        return self.units.energy_unit

    def check_and_set(self, dim_node: int, dim_edge: int, activation: Union[Cell, str] = None):
        """check and set dimension of representation vectors"""
        if self.dim_node is None:
            self.dim_node = get_integer(dim_node)
        elif self.dim_node != dim_node:
            raise ValueError(f'The `dim_node` ({self.dim_node}) of Readout cannot match '
                             f'the dimension of node representation vector ({dim_node}).')

        if self.dim_edge is None:
            self.dim_edge = get_integer(dim_edge)
        elif self.dim_edge != dim_edge:
            raise ValueError(f'The `dim_edge` ({self.dim_edge}) of Readout cannot match '
                             f'the dimension of edge representation vector ({dim_edge}).')

        if activation is not None:
            self.activation = get_activation(activation)

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
        if scale.shape[-1] != self.dim_output and scale.shape[-1] != 1:
            raise ValueError(f'The dimension of "scale" ({scale.shape[-1]}) does not match the '
                             f'output dimension ({self.dim_output})')
        if scale.shape == self.scale.shape:
            F.assign(self.scale, scale)
        else:
            self.scale = Parameter(scale, name='scale', requires_grad=False)

        shift: Tensor = get_ms_array(shift, ms.float32).reshape(-1)
        if shift.shape[-1] != self.dim_output and shift.shape[-1] != 1:
            raise ValueError(f'The dimension of "shift" ({self.shift.shape[-1]}) does not match the '
                             f'output dimension ({self.dim_output})')
        if shift.shape == self.shift.shape:
            F.assign(self.shift, shift)
        else:
            self.shift = Parameter(shift, name='shift', requires_grad=False)

        return self

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  atom_types: Tensor = None,
                  atom_mask: Tensor = None,
                  distances: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  **kwargs,
                  ):
        r"""Compute readout network.

        Args:
            node_rep (Tensor): Tensor of shape (B, A, F). Data type is float.
                Atomic (node) representation vector.
            edge_rep (Tensor): Tensor of shape (B, A, N, F). Data type is float.
                Edge representation vector.
            node_emb (Tensor): Tensor of shape (B, A, F). Data type is float.
                Atomic (node) embedding vector.
            edge_emb (Tensor): Tensor of shape (B, A, F). Data type is float.
                Edge embedding vector.
            atom_types (Tensor): Tensor of shape (B, A). Data type is int.
                Index of atom types. Default: None
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool
                Mask for atom types
            distances (Tensor): Tensor of shape (B, A, N). Data type is float.
                Distances between atoms
            neighbours (Tensor): Tensor of shape (B, A, N). Data type is int.
                Indices of other near neighbour atoms around a atom
            neighbour_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for neighbours

        Returns:
            output: (Tensor):    Tensor of shape (B, A, Y). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of representation.
            Y:  Output dimension.

        """
        raise NotImplementedError


@_readout_register('node')
class NodeReadout(Readout):
    r"""
    Readout function

    Args:

        dim_output (int):           Output dimension.

        activation (Cell):          Activation function

        decoder (str):              Decoder network for atom representation. Default: 'halve'

        aggregator (str):           Aggregator network for atom representation. Default: 'sum'

        scale (float):              Scale value for output. Default: 1

        shift (float):              Shift value for output. Default: 0

        type_ref (Tensor):          Tensor of shape `(T, Y)`. Data type is float.
                                    Reference value for atom types. Default: None

        atomwise_scaleshift (bool): To use atomwise scaleshift (True) or graph scaleshift (False).
                                    Default: False

        axis (int):                 Axis to readout. Default: -2

        n_decoder_layers (list):    number of neurons in each hidden layer of the decoder network.
                                    Default: 1

        energy_unit (str):          Energy unit of output. Default: None

        hyper_param (dict):         Hyperparameter. Default: None

    Symbols:

        B:  Batch size.

        A:  Number of atoms.

        T:  Number of atom types.

        Y:  Output dimension.

    """

    def __init__(self,
                 dim_output: int = 1,
                 dim_node: int = None,
                 activation: Cell = None,
                 decoder: Decoder = 'halve',
                 aggregator: Aggregator = 'default',
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 axis: int = -2,
                 n_decoder_layers: int = 1,
                 mode: str = 'atomwise',
                 energy_unit: str = None,
                 **kwargs,
                 ):
        super().__init__(
            dim_output=dim_output,
            dim_node=dim_node,
            dim_edge=None,
            activation=activation,
            scale=scale,
            shift=shift,
            energy_unit=energy_unit,
            **kwargs,
        )
        self._kwargs = get_arguments(locals())

        self.n_decoder_layers = get_integer(n_decoder_layers)

        if mode.lower() in ['atomwise', 'a']:
            self.atomwise_readout = True
        elif mode.lower() in ['graph', 'set2set', 'g']:
            self.atomwise_readout = True
        else:
            self.atomwise_readout = None
            raise ValueError(f'Unknown mode: {mode}')

        self.decoder = decoder
        if isinstance(decoder, (Decoder, dict)) or self.dim_edge is not None:
            self.decoder = get_decoder(decoder, self.dim_node, self.dim_output,
                                       self.activation, self.n_decoder_layers)
            if self.decoder is None:
                self.dim_node = None
            else:
                self.dim_node = self.decoder.dim_in

        if isinstance(aggregator, str) and aggregator.lower() == 'default':
            aggregator = 'sum' if self.atomwise_readout else 'mean'

        self.aggregator = get_aggregator(aggregator, self.dim_output, axis)

        self.axis = get_integer(axis)

        self.reduce_sum = P.ReduceSum()

        self.type_ref = get_ms_array(type_ref, ms.float32)
        if self.type_ref is not None:
            self.type_ref = Parameter(self.type_ref, name='type_ref', requires_grad=False)

    def check_and_set(self, dim_node: int, dim_edge: int, activation: Cell = None):
        super().check_and_set(dim_node, dim_edge, activation)
        if self.dim_node is not None and isinstance(self.decoder, str):
            self.decoder = get_decoder(self.decoder, self.dim_node, self.dim_output,
                                       self.activation, self.n_decoder_layers)

    def set_type_ref(self, type_ref: Union[Tensor, ndarray]):
        type_ref: Tensor = get_ms_array(type_ref, ms.float32)
        if type_ref is None:
            self.type_ref = None
            return self

        if type_ref.ndim != 2:
            raise ValueError(f'The rank (ndim) of type_ref must be 2 but got: {type_ref.ndim}')
        if type_ref.shape[-1] != self.dim_output and type_ref.shape[-1] != 1:
            raise ValueError(f'The dimension of "type_ref" ({self.type_ref.shape[-1]}) does not match the '
                                f'output dimension ({self.dim_output})')

        if self.type_ref is not None and self.type_ref.shape == type_ref.shape:
            F.assign(self.type_ref, type_ref)
        else:
            self.type_ref = Parameter(self.type_ref, name='type_ref', requires_grad=False)

        return self

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+" Activation function: "+str(self.activation))
        if self.decoder is not None:
            print(ret+gap+" Decoder: "+str(self.decoder.cls_name))
        if self.aggregator is not None:
            print(ret+gap+" Aggregator: "+str(self.aggregator.cls_name))
        print(ret+gap+" Representation dimension: "+str(self.dim_node))
        print(ret+gap+" Readout dimension: "+str(self.dim_output))
        print(ret+gap+" Scale: "+str(self.scale.asnumpy()))
        print(ret+gap+" Shift: "+str(self.shift.asnumpy()))
        if self.type_ref is None:
            print(ret+gap+" Reference value for atom types: None")
        else:
            print(ret+gap+" Reference value for atom types:")
            for i, ref in enumerate(self.type_ref):
                print(ret+gap+gap+' No.{: <5}'.format(str(i)+': ')+str(ref))
        print(ret+gap+" Output unit: "+str(self.units.energy_unit_name))
        print(ret+gap+" Reduce axis: "+str(self.axis))
        print('-'*80)
        return self

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  atom_types: Tensor = None,
                  atom_mask: Tensor = None,
                  distances: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  **kwargs,
                  ):
        r"""Compute readout network.

        Args:
            node_rep (Tensor): Tensor of shape (B, A, F). Data type is float.
                Atomic (node) representation vector.
            edge_rep (Tensor): Tensor of shape (B, A, N, F). Data type is float.
                Edge representation vector.
            node_emb (Tensor): Tensor of shape (B, A, F). Data type is float.
                Atomic (node) embedding vector.
            edge_emb (Tensor): Tensor of shape (B, A, F). Data type is float.
                Edge embedding vector.
            atom_types (Tensor): Tensor of shape (B, A). Data type is int.
                Index of atom types. Default: None
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool
                Mask for atom types
            distances (Tensor): Tensor of shape (B, A, N). Data type is float.
                Distances between atoms
            neighbours (Tensor): Tensor of shape (B, A, N). Data type is int.
                Indices of other near neighbour atoms around a atom
            neighbour_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for neighbours

        Returns:
            output: (Tensor):    Tensor of shape (B, A, Y). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of representation.
            Y:  Output dimension.

        """
        
        if atom_mask is None:
            num_atoms = node_rep.shape[-2]
        else:
            num_atoms = count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        if self.atomwise_readout:
            if self.decoder is not None:
                # (B, A, Y) <- (B, A, F)
                y = self.decoder(node_rep)

            if self.aggregator is not None:
                # (B, A, Y)
                y = y * self.scale + self.shift
                if self.type_ref is not None:
                    y += F.gather(self.type_ref, atom_types, 0)
                # (B, Y) <- (B, A, Y)
                y = self.aggregator(y, atom_mask, num_atoms)

        else:
            y = self.aggregator(node_rep, atom_mask, num_atoms)

            if self.decoder is not None:
                y = self.decoder(y)

            y = y * self.scale + self.shift

            if self.type_ref is not None:
                ref = F.gather(self.type_ref, atom_types, 0)
                y += self.reduce_sum(ref, self.axis)

        return y


_READOUT_BY_NAME = {out.__name__: out for out in _READOUT_BY_KEY.values()}


def get_readout(cls_name: Union[Readout, str],
                dim_output=1,
                dim_node=None,
                dim_edge=None,
                activation=None,
                scale=1,
                shift=0,
                unit=None,
                **kwargs,
                ) -> Readout:
    """get readout function

    Args:
        readout (str):          Name of readout function. Default: None
        model (MolecularGNN): Molecular model. Default: None
        dim_output (int):       Output dimension. Default: 1
        energy_unit (str):      Energy Unit. Default: None

    """
    if isinstance(cls_name, Readout):
        return cls_name
    if cls_name is None:
        return None

    if isinstance(cls_name, dict):
        return get_readout(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _READOUT_BY_KEY.keys():
            return _READOUT_BY_KEY[cls_name.lower()](
                dim_output=dim_output,
                dim_node=dim_node,
                dim_edge=dim_edge,
                activation=activation,
                scale=scale,
                shift=shift,
                unit=unit,
                **kwargs,
            )
        if cls_name in _READOUT_BY_NAME.keys():
            return _READOUT_BY_NAME[cls_name](
                dim_output=dim_output,
                dim_node=dim_node,
                dim_edge=dim_edge,
                activation=activation,
                scale=scale,
                shift=shift,
                unit=unit,
                **kwargs,
            )
        raise ValueError(
            "The Readout corresponding to '{}' was not found.".format(cls_name))
    raise TypeError("Unsupported Readout type '{}'.".format(type(cls_name)))
