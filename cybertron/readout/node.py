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
from mindspore.nn import Cell
from mindspore.numpy import count_nonzero
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindsponge.function import get_integer, get_ms_array, get_arguments

from .readout import Readout, _readout_register
from ..aggregator import NodeAggregator, get_node_aggregator
from ..decoder import Decoder, get_decoder


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
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 activation: Union[Cell, str] = None,
                 decoder: Union[Decoder, dict, str] = 'halve',
                 aggregator: Union[NodeAggregator, dict, str] = 'default',
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 axis: int = -2,
                 mode: str = 'atomwise',
                 energy_unit: str = None,
                 **kwargs,
                 ):
        super().__init__(
            dim_node_rep=dim_node_rep,
            dim_edge_rep=(dim_node_rep if dim_edge_rep is None else dim_edge_rep),
            activation=activation,
            scale=scale,
            shift=shift,
            ndim=1,
            shape=(dim_output,),
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.dim_output = get_integer(dim_output)

        if mode.lower() in ['atomwise', 'a']:
            self.atomwise_readout = True
        elif mode.lower() in ['graph', 'set2set', 'g']:
            self.atomwise_readout = True
        else:
            self.atomwise_readout = None
            raise ValueError(f'Unknown mode: {mode}')

        self.decoder = decoder
        if isinstance(decoder, (Decoder, dict)) or self.dim_edge_rep is not None:
            self.decoder = get_decoder(self.decoder, self.dim_node_rep, self.dim_output,
                                       self.activation)
            if self.decoder is None:
                self.dim_node_rep = None
            else:
                self.dim_node_rep = self.decoder.dim_in

        if aggregator is None and not self.atomwise_readout:
            raise ValueError('The aggreator cannot be None under Graph mode!')
        if isinstance(aggregator, str) and aggregator.lower() == 'default':
            aggregator = 'sum' if self.atomwise_readout else 'mean'

        self.aggregator = get_node_aggregator(aggregator, self.dim_output, axis)

        self.axis = get_integer(axis)

        self.reduce_sum = P.ReduceSum()

        self.type_ref = None
        if type_ref is not None:
            type_ref = get_ms_array(type_ref, ms.float32)
            self.type_ref = Parameter(type_ref, name='type_ref', requires_grad=False)

    def set_dimension(self, dim_node_rep: int, dim_edge_rep: int):
        super().set_dimension(dim_node_rep, dim_edge_rep)
        if self.dim_node_rep is not None and isinstance(self.decoder, str):
            self.decoder = get_decoder(self.decoder, self.dim_node_rep, self.dim_output,
                                       self.activation)

    def set_type_ref(self, type_ref: Union[Tensor, ndarray]):
        """set type reference"""
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
        print(ret+gap+" Representation dimension: "+str(self.dim_node_rep))
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
        print(ret+gap+f" Reduce axis: {self.axis}")
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
            output: (Tensor): Tensor of shape `(B, Y)`. Data type is float

        Symbols:
            B: Batch size.
            A: Number of atoms in system.
            F: Feature dimension of node representation vector.
            G: Feature dimension of edge representation vector.
            E: Feature dimension of node embedding vector.
            K: Feature dimension of edge embedding vector.
            D: Spatial dimension of the simulation system. Usually is 3.
            Y: Output dimension.

        """
        #pylint: disable=unused-argument

        if atom_mask is None:
            num_atoms = node_rep.shape[-2]
        else:
            num_atoms = count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        if self.atomwise_readout:
            y = node_rep
            if self.decoder is not None:
                # (B, A, Y) <- (B, A, F)
                y = self.decoder(node_rep)

            if self.aggregator is not None:
                # (B, A, Y)
                y = y * self.scale + self.shift
                if self.type_ref is not None:
                    y += F.gather(self.type_ref, atom_type, 0)
                # (B, Y) <- (B, A, Y)
                y = self.aggregator(y, atom_mask, num_atoms)

        else:
            y = self.aggregator(node_rep, atom_mask, num_atoms)

            if self.decoder is not None:
                y = self.decoder(y)

            y = y * self.scale + self.shift

            if self.type_ref is not None:
                ref = F.gather(self.type_ref, atom_type, 0)
                y += self.reduce_sum(ref, self.axis)

        return y
