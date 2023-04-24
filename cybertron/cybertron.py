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
Main program of Cybertron
"""

from typing import Union, List, Tuple
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell, CellList, Norm
from mindspore.ops import functional as F
from mindspore import ops

from mindsponge.function import Units, GLOBAL_UNITS
from mindsponge.function import get_integer, get_arguments
from mindsponge.function import GetVector, gather_vector
from mindsponge.partition import FullConnectNeighbours
from mindsponge.potential import PotentialCell

from .embedding import GraphEmbedding, get_embedding
from .readout import Readout, get_readout
from .model import MolecularGNN, get_molecular_model


class Cybertron(Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:

        model (Cell):           Deep molecular model.

        readout (Cell):         Readout function.

        dim_output (int):       Output dimension. Default: 1.

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_type (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: None,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: None.

        num_atom_types (int):   Maximum number of atomic types. Default: 64

        pbc_box (Tensor):       Tensor of shape (B, D).
                                Box size of periodic boundary condition. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Unit of position coordinate. Default: None

        energy_unit (str):      Unit of output energy. Default: None.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: None.

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        O:  Output dimension of the predicted properties.

    """

    def __init__(self,
                 model: Union[MolecularGNN, dict, str],
                 embedding: Union[GraphEmbedding, dict, str] = None,
                 readout: Union[Readout, dict, str, List[Readout]] = 'node',
                 num_atoms: int = None,
                 atom_type: Union[Tensor, ndarray, List[int]] = None,
                 bond_types: Union[Tensor, ndarray, List[int]] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 use_pbc: bool = None,
                 concat_outputs: bool = None,
                 concat_axis: int = -1,
                 length_unit: Union[str, Units] = None,
                 energy_unit: Union[str, Units] = None,
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = get_arguments(locals(), kwargs)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        if energy_unit is None:
            energy_unit = GLOBAL_UNITS.energy_unit
        self._units = Units(length_unit, energy_unit)

        if atom_type is None:
            self.atom_type = None
            self.atom_mask = None
            if num_atoms is None:
                raise ValueError(
                    '"num_atoms" must be assigned when "atom_type" is None')
            natoms = get_integer(num_atoms)
            self.num_atoms = natoms
        else:
            # (1,A)
            self.atom_type = Tensor(atom_type, ms.int32).reshape(1, -1)
            self.atom_mask = atom_type > 0
            natoms = self.atom_type.shape[-1]
            if self.atom_mask.all():
                self.num_atoms = natoms
            else:
                self.num_atoms = F.cast(atom_type > 0, ms.int32)
                self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bond_types = None
        self.bond_mask = None
        if bond_types is not None:
            self.bond_types = Tensor(
                bond_types, ms.int32).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        model = get_molecular_model(model)
        dim_node_emb = model.dim_node_emb
        dim_edge_emb = model.dim_edge_emb
        self.activation = model.activation

        if embedding is None:
            embedding = model.default_embedding

        self.embedding = get_embedding(embedding,
                                       dim_node=dim_node_emb,
                                       dim_edge=dim_edge_emb,
                                       activation=self.activation,
                                       length_unit=length_unit
                                       )

        self.dim_node_emb = self.embedding.dim_node
        self.dim_edge_emb = self.embedding.dim_edge

        model.set_dimension(self.dim_node_emb, self.dim_edge_emb)
        self.model = model

        self.dim_node_rep = self.model.dim_node_rep
        self.dim_edge_rep = self.model.dim_edge_rep

        self.calc_distance = self.embedding.emb_dis

        self.neighbours = None
        self.neighbour_mask = None
        self.get_neigh_list = None
        self.pbc_box = None
        self.use_pbc = use_pbc
        self.num_neighbours = None
        self.cutoff = None
        self.large_dis = 5e4

        if self.calc_distance:
            self.cutoff = self.embedding.cutoff

            self.get_neigh_list = FullConnectNeighbours(natoms)
            self.num_neighbours = self.num_atoms - 1

            if self.atom_type is not None:
                self.neighbours, self.neighbour_mask = self.get_neigh_list(
                    self.atom_type > 0)

            if pbc_box is not None:
                # (1,D)
                self.pbc_box = Tensor(pbc_box, ms.float32).reshape(1, -1)
                self.use_pbc = True

            self.get_vector = GetVector(self.use_pbc)
            self.large_dis = self.cutoff * 10

        self.norm_last_dim = Norm(-1, False)

        self.activation = self.model.activation

        self.num_readouts = 0
        self.num_outputs = 2
        self.output_ndim = (2, 3)
        # [(A, F), (A, N, F)]
        self.output_shape = ((self.num_atoms, self.dim_node_rep),
                             (self.num_atoms, self.num_neighbours, self.dim_edge_rep))
        self.readout = None
        if readout is not None:
            if isinstance(readout, (Readout, str, dict)):
                self.num_readouts = 1
                self.num_outputs = 1
                readout = [readout]
            if isinstance(readout, (list, tuple)):
                self.num_readouts = len(readout)
                self.num_outputs = len(readout)
                readout = [get_readout(cls_name=r,
                                       dim_node_rep=self.dim_node_rep,
                                       dim_edge_rep=self.dim_edge_rep,
                                       activation=self.activation,
                                       ) for r in readout]
            else:
                readout = None
                raise TypeError(f'Unsupported `readout` type: {type(readout)}')

            self.output_ndim = []
            self.output_shape = []
            for i in range(self.num_outputs):
                readout[i].set_dimension(self.dim_node_rep, self.dim_edge_rep)
                self.output_ndim.append(readout[i].ndim)
                self.output_shape.append(readout[i].shape)

            self.readout = CellList(readout)

        self.concat_outputs = concat_outputs
        self.concat_axis = get_integer(concat_axis)

        if self.concat_outputs is not False:
            ndim_set = set(self.output_ndim)
            if self.concat_outputs is None and len(ndim_set) == 1 and (1 in ndim_set):
                self.concat_outputs = True
            if self.concat_outputs and len(ndim_set) > 1:
                raise ValueError(f'Outputs cannot be concatenated when '
                                 f'the ndim of the outputs are different: {self.output_ndim}.')

        if self.concat_outputs:
            rest_shape = []
            cdim = 0
            for shape in self.output_shape:
                shape_ = list((-1,) + shape)
                cdim += shape_[self.concat_axis]
                shape_[self.concat_axis] = -1
                rest_shape.append(tuple(shape_))

            if len(set(rest_shape)) != 1:
                raise ValueError(f'The shape of outputs cannot be concatenated '
                                 f'with axis {self.concat_axis}: {self.output_shape}.')

            output_shape = list(rest_shape[0])
            output_shape[self.concat_axis] = cdim
            output_shape = tuple(output_shape[0:])
            self.output_shape = (output_shape,)
            self.output_ndim = (len(output_shape),)
            self.num_outputs = 1

        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        self.output_unit_scale = [Tensor(self.readout[i].convert_energy_to(self._units), ms.float32)
                                  for i in range(self.num_readouts)]

        self.concat = ops.Concat(self.concat_axis)

    @property
    def units(self) -> Units:
        return self._units

    @units.setter
    def units(self, units_: Units):
        self._units = units_
        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        if self.readout is not None:
            self.output_unit_scale = \
                (Tensor(self.readout[i].convert_energy_to(self._units), ms.float32)
                 for i in range(self.num_readouts))
    @property
    def length_unit(self) -> str:
        return self._units.length_unit

    @length_unit.setter
    def length_unit(self, length_unit_: Union[str, Units]):
        self.set_length_unit(length_unit_)

    @property
    def energy_unit(self) -> str:
        return self._units.energy_unit

    @energy_unit.setter
    def energy_unit(self, energy_unit_: Union[str, Units]):
        self.set_energy_unit(energy_unit_)

    @property
    def model_name(self) -> str:
        return self.model.cls_name

    def readout_ndim(self, idx: int) -> int:
        if idx >= self.num_readouts:
            raise ValueError(f'The index ({idx}) is exceed '
                             f'the number of readout ({self.num_readouts})')
        return self.readout[idx].ndim

    def readout_shape(self, idx: int) -> Tuple[int]:
        if idx >= self.num_readouts:
            raise ValueError(f'The index ({idx}) is exceed '
                             f'the number of readout ({self.num_readouts})')
        return self.readout[idx].shape

    def set_units(self, length_unit: str = None, energy_unit: str = None):
        """set units"""
        if length_unit is not None:
            self.set_length_unit(length_unit)
        if energy_unit is not None:
            self.set_energy_unit(energy_unit)
        return self

    def set_length_unit(self, length_unit: str):
        """set length unit"""
        self._units = self._units.set_length_unit(length_unit)
        self.input_unit_scale = self.embedding.convert_length_from(self._units)
        return self

    def set_energy_unit(self, energy_units: str):
        """set energy unit"""
        self._units.set_energy_unit(energy_units)
        if self.readout is not None:
            self.output_unit_scale = (
                Tensor(self.readout[i].convert_energy_to(self._units), ms.float32)
                for i in range(self.num_readouts))
        return self

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of Cybertron"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('-'*80)
        if self.atom_type is None:
            print(f'{ret} Using variable atom types with maximum number of atoms: {self.num_atoms}')
        else:
            print(f'{ret} Using fixed atom type index:')
            for i, atom in enumerate(self.atom_type[0]):
                print(ret+gap+' Atom {: <7}'.format(str(i))+f': {atom.asnumpy()}')
        if self.bond_types is not None:
            print(ret+' Using fixed bond connection:')
            for b in self.bond_types[0]:
                print(ret+gap+' '+str(b.asnumpy()))
            print(ret+' Fixed bond mask:')
            for m in self.bond_mask[0]:
                print(ret+gap+' '+str(m.asnumpy()))
        print('-'*80)
        self.embedding.print_info(num_retraction=num_retraction,
                                  num_gap=num_gap, char=char)
        self.model.print_info(num_retraction=num_retraction,
                              num_gap=num_gap, char=char)

        print(ret+" With "+str(self.num_readouts)+" readout networks: ")
        print('-'*80)
        for i in range(self.num_readouts):
            print(ret+" "+str(i)+(". "+self.readout[i].cls_name))
            self.readout[i].print_info(
                num_retraction=num_retraction, num_gap=num_gap, char=char)
            
        print(f'{ret} Input unit: {self._units.length_unit_name}')
        print(f'{ret} Output unit: {self._units.energy_unit_name}')
        print(f'{ret} Input unit scale: {self.input_unit_scale}')
        print(f'{ret} output unit scale:')
        for i in range(self.num_readouts):
            print(ret+gap+f" Readout {i}: {self.output_unit_scale[i]}")
        print("================================================================================")

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       unit: str = None,
                       readout_id: int = None,
                       ):
        """set the scale and shift"""
        
        def _set_scaleshift(scale_: float = 1,
                            shift_: float = 0,
                            unit_: str = None,
                            idx: int = None,
                            ):
            self.readout[idx].set_scaleshift(scale=scale_, shift=shift_, unit=unit_)
            self.output_unit_scale[idx] = \
                Tensor(self.readout[idx].convert_energy_to(self._units), ms.float32)

        if readout_id is None:
            for i in range(self.num_readouts):
                _set_scaleshift(scale, shift, unit, i)
        else:
            if readout_id >= self.num_readouts:
                raise ValueError(f'The index ({readout_id}) exceeds '
                                 f'the number of readouts ({self.num_readouts})')
            _set_scaleshift(scale, shift, unit, readout_id)

        return self

    def construct(self,
                  coordinate: Tensor = None,
                  atom_type: Tensor = None,
                  pbc_box: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  ):
        """Compute the properties of the molecules.

        Args:
            coordinate (Tensor): Tensor of shape (B, A, D). Data type is float.
                Cartesian coordinates for each atom.
            atom_type (Tensor): Tensor of shape (B, A). Data type is int.
                Type index (atomic number) of atom types.
            pbc_box (Tensor): Tensor of shape (B, D). Data type is float.
                Box size of periodic boundary condition
            distance (Tensor): Tensor of shape (B, A, N). Data type is float.
                Distances between atoms
            neighbours (Tensor): Tensor of shape (B, A, N). Data type is int.
                Indices of other near neighbour atoms around a atom
            neighbour_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for neighbours
            bond_types (Tensor): Tensor of shape (B, A, N). Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape (B, A, N). Data type is bool.
                Mask for bonds

        Returns:
            outputs (Tensor):    Tensor of shape (B, A, O). Data type is float.

        """

        if self.atom_type is None:
            # (1, A)
            atom_mask = atom_type > 0
        else:
            # (1, A)
            atom_type = self.atom_type
            atom_mask = self.atom_mask

        neigh_dis = None
        neigh_vec = None
        if coordinate is not None:
            coordinate *= self.input_unit_scale
            if neighbours is None:
                neighbours = self.neighbours
                neighbour_mask = self.neighbour_mask
                if self.atom_type is None:
                    neighbours, neighbour_mask = self.get_neigh_list(atom_mask)
            if self.pbc_box is not None:
                pbc_box = self.pbc_box

            # (B, A, N, D) <- (B, A, N, D)
            neigh_pos = gather_vector(coordinate, neighbours)
            # (B, A, N, D) = (B, A, N, D) - (B, A, 1, D)
            neigh_vec = self.get_vector(F.expand_dims(coordinate, -2), neigh_pos, pbc_box)

            # Add a non-zero value to the neighbour_vector whose mask value is False
            # to prevent them from becoming zero values after Norm operation,
            # which could lead to auto-differentiation errors
            if neighbour_mask is not None:
                # (B, A, N)
                large_dis = msnp.broadcast_to(self.large_dis, neighbour_mask.shape)
                large_dis = F.select(neighbour_mask, F.zeros_like(large_dis), large_dis)
                # (B, A, N, D) = (B, A, N, D) + (B, A, N, 1)
                neigh_vec += F.expand_dims(large_dis, -1)

            neigh_dis = self.norm_last_dim(neigh_vec)

        node_emb, node_mask, edge_emb, \
            edge_mask, edge_cutoff, edge_self = self.embedding(atom_type=atom_type,
                                                               atom_mask=atom_mask,
                                                               neigh_dis=neigh_dis,
                                                               neigh_vec=neigh_vec,
                                                               neigh_list=neighbours,
                                                               neigh_mask=neighbour_mask,
                                                               bond=bonds,
                                                               bond_mask=bond_mask,
                                                               )

        node_rep, edge_rep = self.model(node_emb=node_emb,
                                        node_mask=node_mask,
                                        neigh_list=neighbours,
                                        edge_emb=edge_emb,
                                        edge_mask=edge_mask,
                                        edge_cutoff=edge_cutoff,
                                        edge_self=edge_self,
                                        )

        if self.readout is None:
            return node_rep, node_rep

        outputs = ()
        for i in range(self.num_readouts):
            outputs += (self.readout[i](node_rep=node_rep,
                                        edge_rep=edge_rep,
                                        node_emb=node_emb,
                                        atom_type=atom_type,
                                        atom_mask=atom_mask,
                                        neigh_dis=neigh_dis,
                                        neigh_vec=neigh_vec,
                                        neigh_list=neighbours,
                                        neigh_mask=neighbour_mask,
                                        bond=bonds,
                                        bond_mask=bond_mask,
                                        ) * self.output_unit_scale[i],
                        )

        if self.concat_outputs:
            outputs = self.concat(outputs)
        elif self.num_readouts == 1:
            outputs = outputs[0]

        return outputs


class CybertronFF(PotentialCell):
    """Cybertron as potential for Mindmindsponge.

    Args:

        model (Cell):           Deep molecular model. Default: None

        readout (Cell):         Readout function. Default: 'atomwise'

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_type (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: None,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: None.

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Unit of position coordinate. Default: None

        energy_unit (str):      Unit of output energy. Default: None.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: None.

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

    """
    def __init__(self,
                 model: MolecularGNN = None,
                 readout: Readout = 'atomwise',
                 num_atoms: int = None,
                 atom_type: Tensor = None,
                 bond_types: Tensor = None,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 **kwargs
                 ):

        super().__init__(
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        dim_output = 1
        pbc_box = None

        if dim_output != 1:
            raise ValueError('The output dimension of CybertronFF must be 1 but got: '+str(dim_output))
        if readout is None:
            raise ValueError('The readout function in CybertronFF cannot be None!')

        self.model = get_molecular_model(model, length_unit=self.length_unit)

        self.dim_feature = self.model.dim_feature
        self.activation = self.model.activation
        self.input_unit_scale = self._units.convert_energy_to(self.model.units)

        if isinstance(readout, (list, tuple)):
            raise ValueError('CybertronFF cannot accept multiple readouts!')

        self.readout = get_readout(
            readout,
            model=self.model,
            dim_output=1,
            energy_unit=self._units.energy_unit,
        )
        if self.readout.dim_output != 1:
            raise ValueError('The output dimension of readout in CybertronFF must be 1 but got: '+
                             str(self.readout.dim_output))
        self.dim_output = self.readout.dim_output

        self.atomwise_scaleshift = self.readout.atomwise_scaleshift
        self.output_unit_scale = self.get_output_unit_scale()

        if atom_type is None:
            raise ValueError('For CybertronFF, atom_type cannot be None')

        # (1,A)
        self.atom_type = Tensor(atom_type, ms.int32).reshape(1, -1)
        self.atom_mask = self.atom_type > 0
        natoms = self.atom_type.shape[-1]
        if self.atom_mask.all():
            self.num_atoms = natoms
        else:
            self.num_atoms = F.cast(atom_type > 0, ms.int32)
            self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bond_types = None
        self.bond_mask = None
        if bond_types is not None:
            self.bond_types = Tensor(bond_types, ms.int32).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        self.pbc_box = None
        if pbc_box is not None:
            # (1,D)
            self.pbc_box = Tensor(pbc_box, ms.float32).reshape(1, -1)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is None:
            if length_unit is not None:
                self._units.set_length_unit(length_unit)
            if energy_unit is not None:
                self._units.set_energy_unit(energy_unit)
        else:
            self._units.set_units(units=units)
            self.length_unit = self._units.length_unit
            self.energy_unit = self._units.energy_unit
            self.input_unit_scale = self._units.convert_energy_to(
                self.model.units)
            self.output_unit_scale = self.get_output_unit_scale()
        return self

    def get_output_unit_scale(self) -> Tensor:
        """get the scale factor of output unit"""
        output_unit_scale = self._units.convert_energy_from(
            self.readout.energy_unit)
        return Tensor(output_unit_scale, ms.float32)

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of CybertronFF"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Force Field:")
        print('-'*80)
        print(ret+' Length unit: ' + self._units.length_unit_name)
        print(ret+' Input unit scale: ' + str(self.input_unit_scale))
        for i, atom in enumerate(self.atom_type[0]):
            print(
                ret+gap+' Atom {: <7}'.format(str(i)+': ')+str(atom.asnumpy()))
        if self.bond_types is not None:
            print(ret+' Using fixed bond connection:')
            for b in self.bond_types[0]:
                print(ret+gap+' '+str(b.asnumpy()))
            print(ret+' Fixed bond mask:')
            for m in self.bond_mask[0]:
                print(ret+gap+' '+str(m.asnumpy()))
        print('-'*80)
        self.model.print_info(num_retraction=num_retraction,
                              num_gap=num_gap, char=char)

        print(ret+" Readout network: "+self.readout.cls_name)
        print('-'*80)
        self.readout.print_info(
            num_retraction=num_retraction, num_gap=num_gap, char=char)
        print(ret+" Output unit for Cybertron: "+self._units.energy_unit_name)
        print(ret+" Output unit scale: "+str(self.output_unit_scale))
        print("================================================================================")

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       type_ref: Tensor = None,
                       atomwise_scaleshift: bool = None,
                       unit: str = None,
                       ):
        """set the scale and shift"""
        self.readout.set_scaleshift(
            scale=scale, shift=shift, type_ref=type_ref, atomwise_scaleshift=atomwise_scaleshift, unit=unit)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinate (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor): Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        #pylint: disable=unused-argument

        x, xlist = self.model(neighbour_distance, self.atom_type, self.atom_mask,
                              neighbour_index, neighbour_mask)

        energy = self.readout(x, xlist, self.atom_type, self.atom_mask, self.num_atoms)

        return energy * self.output_unit_scale
