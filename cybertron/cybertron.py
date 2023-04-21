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
from mindspore.nn import Cell, CellList
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindsponge.function import concat_last_dim
from mindsponge.function import Units, GLOBAL_UNITS
from mindsponge.function import get_integer, get_ms_array, get_arguments
from mindsponge.partition import IndexDistances
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

        atom_types (Tensor):    Tensor of shape (B, A). Data type is int.
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
                 embedding: Union[GraphEmbedding, dict, str] = 'molecule',
                 readout: Union[Readout, dict, str, List[Readout]] = 'node',
                 num_atoms: int = None,
                 atom_types: Union[Tensor, ndarray, List[int]] = None,
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

        if atom_types is None:
            self.atom_types = None
            self.atom_mask = None
            if num_atoms is None:
                raise ValueError(
                    '"num_atoms" must be assigned when "atom_types" is None')
            natoms = get_integer(num_atoms)
            self.num_atoms = natoms
        else:
            # (1,A)
            self.atom_types = Tensor(atom_types, ms.int32).reshape(1, -1)
            self.atom_mask = atom_types > 0
            natoms = self.atom_types.shape[-1]
            if self.atom_mask.all():
                self.num_atoms = natoms
            else:
                self.num_atoms = F.cast(atom_types > 0, ms.int32)
                self.num_atoms = msnp.sum(num_atoms, -1, keepdims=True)

        self.bond_types = None
        self.bond_mask = None
        if bond_types is not None:
            self.bond_types = Tensor(
                bond_types, ms.int32).reshape(1, natoms, -1)
            self.bond_mask = bond_types > 0

        self.fc_neighbours = FullConnectNeighbours(natoms)
        self.neighbours = None
        self.neighbour_mask = None
        self.num_neighbours = self.num_atoms - 1
        if self.atom_types is not None:
            self.neighbours, self.neighbour_mask = self.fc_neighbours(
                self.atom_types > 0)
            
        self.pbc_box = None
        self.use_pbc = use_pbc
        if pbc_box is not None:
            # (1,D)
            self.pbc_box = Tensor(pbc_box, ms.float32).reshape(1, -1)
            self.use_pbc = True

        self.model = get_molecular_model(model, length_unit=self.length_unit)
        dim_node_emb = self.model.dim_node_emb
        dim_edge_emb = self.model.dim_edge_emb
        self.activation = self.model.activation

        self.embedding = get_embedding(embedding,
                                       dim_node=dim_node_emb,
                                       dim_edge=dim_edge_emb,
                                       activation=self.activation,
                                       length_unit=length_unit
                                       )
        self.dim_node_emb = self.embedding.dim_node
        self.dim_edge_emb = self.embedding.dim_edge

        self.model.set_dimension(self.dim_node_emb, self.dim_edge_emb)

        self.dim_node_rep = self.model.dim_node_rep
        self.dim_edge_rep = self.model.dim_edge_rep

        self.activation = self.model.activation

        self.num_readouts = 0
        self.num_outputs = 2
        self.output_ndim = (2, 3)
        # [(A, F), (A, N, F)]
        self.output_shape = ((self.num_atoms, self.dim_node_rep),
                             (self.num_atoms, self.num_neighbours, self.dim_edge_rep))
        if readout is not None:
            if isinstance(readout, Readout):
                readout = [readout]
            if isinstance(readout, (list, tuple)):
                readout = CellList(readout)
            if isinstance(readout, CellList):
                self.num_readouts = len(readout)
                self.num_outputs = len(readout)
                self.readout = readout
            else:
                raise TypeError(f'Unsupported `readout` type: {type(readout)}')

            self.output_ndim = []
            self.output_shape = []
            for i in range(self.num_outputs):
                self.output_ndim.append(self.readout[i].ndim)
                self.output_shape.append(self.readout[i].shape)

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
                rest_shape.append(shape_)
            if len(set(rest_shape)) != 1:
                raise ValueError(f'The shape of outputs cannot be concatenated '
                                 f'with axis {self.concat_axis}: {self.output_shape}.')
            
            output_shape = rest_shape[0]
            output_shape[self.concat_axis] = cdim
            output_shape = tuple(output_shape[0:])
            self.output_shape = (output_shape,)
            self.output_ndim = (len(output_shape),)
            self.num_outputs = 1

        self.input_unit_scale = self.embedding.convert_length_from(self.units)
        self.output_unit_scale = [Tensor(self.readout[i].convert_length_to(self.units), ms.float32)
                                  for i in range(self.num_readouts)]

        self.concat = P.Concat(self.concat_axis)

    @property
    def length_unit(self) -> str:
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        return self.units.energy_unit
    
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

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is None:
            if length_unit is not None:
                self.set_length_unit(length_unit)
            if energy_unit is not None:
                self.set_energy_unit(energy_unit)
        else:
            self.units = units
            self.input_unit_scale = self.embedding.convert_length_from(self.units)
            self.output_unit_scale = \
                (Tensor(self.readout[i].convert_length_to(self.units), ms.float32)
                 for i in range(self.num_readouts))
        return self

    def set_length_unit(self, length_unit: str):
        """set length unit"""
        self.units = self.units.set_length_unit(length_unit)
        self.input_unit_scale = self.embedding.convert_length_from(self.units)
        return self

    def set_energy_unit(self, energy_units: str):
        """set energy unit"""
        self.units.set_energy_unit(energy_units)
        self.output_unit_scale = (Tensor(self.readout[i].convert_length_to(self.units), ms.float32)
                                for i in range(self.num_readouts))
        return self

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of Cybertron"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('-'*80)
        print(f'{ret} Length unit: {self.units.length_unit_name}')
        print(f'{ret} Input unit scale: {self.input_unit_scale}')
        if self.atom_types is not None:
            print(f'{ret} Using fixed atom type index:')
            for i, atom in enumerate(self.atom_types[0]):
                print(ret+gap+' Atom {: <7}'.format(str(i))+f': {atom.asnumpy()}')
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

        print(ret+" With "+str(self.num_readouts)+" readout networks: ")
        print('-'*80)
        for i in range(self.num_readouts):
            print(ret+" "+str(i)+(". "+self.readout[i].cls_name))
            self.readout[i].print_info(
                num_retraction=num_retraction, num_gap=num_gap, char=char)
       
        print(ret+" Output unit for Cybertron: "+self.units.energy_unit_name)
        print(ret+" Output unit scale: "+str(self.output_unit_scale))
        print("================================================================================")

    def set_scaleshift(self,
                       scale: float = 1,
                       shift: float = 0,
                       unit: str = None,
                       readout_id: int = None
                       ):
        """set the scale and shift"""

        self.readout[readout_id].set_scaleshift(scale=scale, shift=shift, unit=unit)
        self.output_unit_scale[readout_id] = \
            Tensor(self.readout[readout_id].convert_length_to(self.units), ms.float32)
        return self

    def construct(self,
                  positions: Tensor = None,
                  atom_types: Tensor = None,
                  pbc_box: Tensor = None,
                  distances: Tensor = None,
                  neighbours: Tensor = None,
                  neighbour_mask: Tensor = None,
                  bonds: Tensor = None,
                  bond_mask: Tensor = None,
                  ):
        """Compute the properties of the molecules.

        Args:
            positions (Tensor):         Tensor of shape (B, A, D). Data type is float.
                                        Cartesian coordinates for each atom.
            atom_types (Tensor):        Tensor of shape (B, A). Data type is int.
                                        Type index (atomic number) of atom types.
                                        Default: self.atom_types
            pbc_box (Tensor):           Tensor of shape (B, D). Data type is float.
                                        Box size of periodic boundary condition
            distances (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                        Distances between atoms
            neighbours (Tensor):        Tensor of shape (B, A, N). Data type is int.
                                        Indices of other near neighbour atoms around a atom
            neighbour_mask (Tensor):    Tensor of shape (B, A, N). Data type is bool.
                                        Mask for neighbours
            bond_types (Tensor):        Tensor of shape (B, A, N). Data type is int.
                                        Types index of bond connected with two atoms
            bond_mask (Tensor):         Tensor of shape (B, A, N). Data type is bool.
                                        Mask for bonds

        Returns:
            properties (Tensor):    Tensor of shape (B, A, O). Data type is float.

        """

        if self.atom_types is None:
            # (1,A)
            atom_mask = atom_types > 0
        else:
            # (1,A)
            atom_types = self.atom_types
            atom_mask = self.atom_mask

        if positions is not None and distance is None:
            if neighbours is None:
                if self.atom_types is None:
                    neighbours, neighbour_mask = self.fc_neighbours(atom_mask)
                else:
                    neighbours = self.neighbours
                    neighbour_mask = self.neighbour_mask
            if self.pbc_box is not None:
                pbc_box = self.pbc_box
            distance = self.get_distance(
                positions, neighbours, neighbour_mask, pbc_box) * self.input_unit_scale

        node_emb, node_mask, edge_emb, \
            edge_mask, edge_cutoff, edge_self = self.embedding(atom_types=atom_types,
                                                               atom_mask=atom_mask,
                                                               neigh_list=neighbours,
                                                               distance=distance,
                                                               dis_mask=neighbour_mask,
                                                               bond=bonds,
                                                               bond_mask=bond_mask)

        node_rep, edge_rep = self.model(node_emb=node_emb,
                                        node_mask=node_mask,
                                        neigh_list=neighbours,
                                        edge_emb=edge_emb,
                                        edge_mask=edge_mask,
                                        edge_cutoff=edge_cutoff,
                                        edge_self=edge_self)

        if self.readout is None:
            return node_rep, node_rep
    
        outputs = (self.readout[i](node_rep=node_rep,
                                   edge_rep=edge_rep,
                                   node_emb=node_emb,
                                   atom_types=atom_types,
                                   atom_mask=atom_mask,
                                   distances=distances,
                                   neighbours=neighbours,
                                   neighbour_mask=neighbour_mask,
                                   ) * self.output_unit_scale[i]
                                   for i in range(self.num_readouts))

        if self.concat_outputs:
            outputs = self.concat(outputs)
        
        return outputs


class CybertronFF(PotentialCell):
    """Cybertron as potential for Mindmindsponge.

    Args:

        model (Cell):           Deep molecular model. Default: None

        readout (Cell):         Readout function. Default: 'atomwise'

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_types (Tensor):    Tensor of shape (B, A). Data type is int.
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

        O:  Output dimension of the predicted properties.

    """
    def __init__(self,
                 model: MolecularGNN = None,
                 readout: Readout = 'atomwise',
                 num_atoms: int = None,
                 atom_types: Tensor = None,
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
        self.input_unit_scale = self.units.convert_energy_to(self.model.units)

        if isinstance(readout, (list, tuple)):
            raise ValueError('CybertronFF cannot accept multiple readouts!')

        self.readout = get_readout(
            readout,
            model=self.model,
            dim_output=1,
            energy_unit=self.units.energy_unit,
        )
        if self.readout.dim_output != 1:
            raise ValueError('The output dimension of readout in CybertronFF must be 1 but got: '+
                             str(self.readout.dim_output))
        self.dim_output = self.readout.dim_output

        self.atomwise_scaleshift = self.readout.atomwise_scaleshift
        self.output_unit_scale = self.get_output_unit_scale()

        if atom_types is None:
            raise ValueError('For CybertronFF, atom_types cannot be None')

        # (1,A)
        self.atom_types = Tensor(atom_types, ms.int32).reshape(1, -1)
        self.atom_mask = self.atom_types > 0
        natoms = self.atom_types.shape[-1]
        if self.atom_mask.all():
            self.num_atoms = natoms
        else:
            self.num_atoms = F.cast(atom_types > 0, ms.int32)
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
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)
        else:
            self.units.set_units(units=units)
            self.length_unit = self.units.length_unit
            self.energy_unit = self.units.energy_unit
            self.input_unit_scale = self.units.convert_energy_to(
                self.model.units)
            self.output_unit_scale = self.get_output_unit_scale()
        return self

    def get_output_unit_scale(self) -> Tensor:
        """get the scale factor of output unit"""
        output_unit_scale = self.units.convert_energy_from(
            self.readout.energy_unit)
        return Tensor(output_unit_scale, ms.float32)

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of CybertronFF"""
        ret = char * num_retraction
        gap = char * num_gap
        print("================================================================================")
        print("Cybertron Force Field:")
        print('-'*80)
        print(ret+' Length unit: ' + self.units.length_unit_name)
        print(ret+' Input unit scale: ' + str(self.input_unit_scale))
        for i, atom in enumerate(self.atom_types[0]):
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
        print(ret+" Output unit for Cybertron: "+self.units.energy_unit_name)
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
        self.atomwise_scaleshift = self.readout.atomwise_scaleshift

        if unit is not None:
            self.units.set_energy_unit(unit)
            self.output_unit_scale = self.get_output_unit_scale()

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

        x, xlist = self.model(neighbour_distance, self.atom_types, self.atom_mask,
                              neighbour_index, neighbour_mask)

        energy = self.readout(x, xlist, self.atom_types, self.atom_mask, self.num_atoms)

        return energy * self.output_unit_scale
