import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from cybertroncode.units import units
from cybertroncode.base import Types2FullConnectNeighbors
from cybertroncode.readouts import Readout,LongeRangeReadout
from cybertroncode.readouts import AtomwiseReadout,GraphReadout
from cybertroncode.neighbors import Distances
        
class Cybertron(nn.Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:
        model       (nn.Cell):          Deep molecular model
        dim_output  (int):              Output dimension of the predictions
        unit_dis    (str):              Unit of input distance
        unit_energy (str):              Unit of output energy
        readout     (readouts.Readout): Readout function

    """

    def __init__(
        self,
        model,
        dim_output=1,
        unit_dis='nm',
        unit_energy=None,
        readout='atomwise',
        max_nodes_number=0,
        node_types=None,
        bond_types=None,
        full_connect=False,
        cut_shape=False,
    ):
        super().__init__()
        
        self.model = model
        self.dim_output = dim_output
        self.full_connect=full_connect
        self.cut_shape = cut_shape

        self.unit_dis = unit_dis
        self.unit_energy = unit_energy

        self.dis_scale = units.length_convert_from(unit_dis)
        activation = self.model.activation
        
        self.molsum = P.ReduceSum(keep_dims=True)

        self.node_mask = None
        if node_types is None:
            self.fixed_atoms=False
            self.num_atoms=0
        else:
            self.fixed_atoms=True
            self.model._set_fixed_atoms(True)

            if len(node_types.shape) == 1:
                self.num_atoms=len(node_types)
            elif len(node_types.shape) == 2:
                self.num_atoms=len(node_types[0])

            if self.num_atoms <= 0:
                raise ValueError("The 'num_atoms' cannot be 0 "+
                    "'node_types' is not 'None' in MolCalculator!")

            if type(node_types) is not Tensor:
                node_types = Tensor(node_types,ms.int32)

            self.node_types = node_types
            self.node_mask = F.expand_dims(node_types,-1) > 0
            if self.node_mask.all():
                self.node_mask = None

            nodes_number = F.cast(node_types>0,ms.float32)
            self.nodes_number = self.molsum(nodes_number,-1)

        self.use_bonds = self.model.use_bonds
        self.fixed_bonds = False
        self.bonds = None
        if bond_types is not None:
            self.bonds = bond_types
            self.bond_mask = (bond_types > 0)
            self.fixed_bonds = True

        self.cutoff = self.model.cutoff

        self.use_distances = self.model.use_distances

        if self.fixed_bonds and (not self.use_distances):
            raise ValueError('"fixed_bonds" cannot be used without using distances')

        self.neighbors = None
        self.mask = None
        self.fc_neighbors = None
        if self.full_connect:
            if self.fixed_atoms:
                self.fc_neighbors = Types2FullConnectNeighbors(self.num_atoms)
                self.neighbors = self.fc_neighbors.get_full_neighbors()
            else:
                if max_nodes_number <= 0:
                    raise ValueError("The 'max_atoms_num' cannot be 0 "+
                        "when the 'full_connect' flag is 'True' and " +
                        "'node_types' is 'None' in MolCalculator!")
                self.fc_neighbors = Types2FullConnectNeighbors(max_nodes_number)

        if self.fixed_atoms and self.full_connect:
            fixed_neigh = True
            self.distances = Distances(True)
            self.model.set_fixed_neighbors(True)
        else:
            fixed_neigh = False
            self.distances = Distances(False)
        self.fixed_neigh = fixed_neigh

        self.multi_readouts = False
        self.num_readout = 1

        dim_feature = self.model.dim_feature
        n_interactions = self.model.n_interactions

        if isinstance(readout,(tuple,list)):
            self.num_readout = len(readout)
            if self.num_readout == 1:
                readout = readout[0]
            else:
                self.multi_readouts = True

        if self.multi_readouts:
            readouts = []
            for i in range(self.num_readout):
                readouts.append(self._get_readout(readout[i],
                    n_in=dim_feature,
                    n_out=dim_output,
                    activation=activation,
                    unit_energy=unit_energy,
                ))
            self.readout = nn.CellList(readouts)
        else:
            self.readout = self._get_readout(readout,
                n_in=dim_feature,
                n_out=dim_output,
                activation=activation,
                unit_energy=unit_energy,
            )

        self.output_scale = 1
        self.calc_far = False
        read_all_interactions = False
        self.dim_output = 0
        if self.multi_readouts:
            read_all_interactions = False
            self.output_scale = []
            for i in range(self.num_readout):
                self.dim_output += self.readout[i].total_out
                if  unit_energy is not None and self.readout[i].output_is_energy:
                    unit_energy = units.check_energy_unit(unit_energy)
                    self.output_scale.append(units.energy_convert_to(unit_energy))
                else:
                    self.output_scale.append(1)

                if isinstance(self.readout[i],LongeRangeReadout):
                    self.calc_far = True
                    self.readout[i].set_fixed_neighbors(fixed_neigh)
                if self.readout[i].read_all_interactions:
                    read_all_interactions = False
                    if self.readout[i].interaction_decoders is not None and self.readout[i].n_interactions != n_interactions:
                        raise ValueError('The n_interactions in model readouts are not equal')
                if self.readout[i].n_in != dim_feature:
                    raise ValueError('n_in in readouts is not equal to dim_feature')
        else:
            self.dim_output = self.readout.total_out

            if unit_energy is not None and self.readout.output_is_energy:
                unit_energy = units.check_energy_unit(unit_energy)
                self.output_scale = units.energy_convert_to(unit_energy)
            else:
                self.output_scale = 1

            if isinstance(self.readout,LongeRangeReadout):
                self.calc_far = True
                self.readout.set_fixed_neighbors(fixed_neigh)

            if self.readout.read_all_interactions:
                read_all_interactions = True
                if self.readout.interaction_decoders is not None and self.readout.n_interactions != n_interactions:
                    raise ValueError('The n_interactions in model readouts are not equal')

            if self.readout.n_in != dim_feature:
                raise ValueError('n_in in readouts is not equal to dim_feature')

        self.unit_energy = unit_energy

        self.model.read_all_interactions = read_all_interactions

        self.ones = P.Ones()
        self.reduceany = P.ReduceAny(keep_dims=True)
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.reducemax = P.ReduceMax()
        self.reducemean = P.ReduceMean(keep_dims=False)
        self.concat = P.Concat(-1)

    def _get_readout(self,
        readout,
        n_in,
        n_out,
        activation,
        unit_energy,
    ):
        if isinstance(readout,Readout):
            return readout
        elif isinstance(readout,str):
            if readout.lower() == 'atom' or readout.lower() == 'atomwise':
                readout = AtomwiseReadout
            elif readout.lower() == 'graph' or readout.lower() == 'set2set':
                readout = GraphReadout
            else:
                raise ValueError("Unsupported Readout type"+readout.lower())

            return readout(
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                unit_energy=unit_energy,
            )

        else:
            raise TypeError("Unsupported Readout type '{}'.".format(type(readout)))

    def print_info(self):
        print("================================================================================")
        print("Cybertron Engine, Ride-on!")
        print('---with input distance unit: '+self.unit_dis)
        print('---with input distance unit: '+self.unit_dis)
        if self.fixed_atoms:
            print('---with fixed atoms: '+str(self.node_types[0]))
        if self.full_connect:
            print('---using full connected neighbors')
        if self.use_bonds and self.fixed_bonds:
            print('---using fixed bond connection:')
            for b in self.bonds[0]:
                print('------'+str(b.asnumpy()))
            print('---with fixed bond mask:')
            for m in self.bond_mask[0]:
                print('------'+str(m.asnumpy()))
        self.model.print_info()

        if self.multi_readouts:
            print("---with multiple readouts: ")
            for i in range(self.num_readout):
                print("---"+str(i+1)+(". "+self.readout[i].name+" readout"))
        else:
            print("---with readout type: "+self.readout.name)
            self.readout.print_info()

        if self.unit_energy is not None:
            print("---with output units: "+str(self.unit_energy))
            print("---with output scale: "+str(self.output_scale))
        print("---with total output dimension: "+str(self.dim_output))
        print("================================================================================")

    def construct(self,
            positions=None,
            node_types=None,
            atom_types=None,
            neighbors=None,
            neighbor_mask=None,
            bonds=None,
            bond_mask=None,
            far_neighbors=None,
            far_mask=None,
        ):
        """Compute the properties of the molecules.

        Args:
            positions     (mindspore.Tensor[float], [B, A, 3]): Cartesian coordinates for each node.
            node_types    (mindspore.Tensor[int],   [B, A]):    Types (ID) of input nodes.
                                                                If the attribute "self.node_types" have been set and
                                                                node_types is not given here, node_types = self.node_type
            atom_types    (mindspore.Tensor[int],   [B, A']):   Types (ID) of the real atoms represented by the input nodes.
                                                                Used to calculate the real number of each type of atoms with atom_ref.
                                                                If atom_types is not given here, atom_types = node_types
            neighbors     (mindspore.Tensor[int],   [B, A, N]): Indices of other near neighbor nodes around a node
            neighbor_mask (mindspore.Tensor[bool],  [B, A, N]): Mask for neighbors
            bonds         (mindspore.Tensor[int],   [B, A, N]): Types (ID) of bond connected with two nodes
            bond_mask     (mindspore.Tensor[bool],  [B, A, N]): Mask for bonds
            far_neighbors (mindspore.Tensor[int],   [B, A, N]): Indices of other far neighbor nodes around a node
            far_mask      (mindspore.Tensor[bool],  [B, A, N]): Mask for far_neighbors
            
            B:  Batch size, usually the number of input molecules or frames
            A:  Number of input node, usually the number of atoms in one molecule or frame
            A': Number of the real atoms in one molecule or frame. If all the atoms are represented by input nodes, A = A'.
            N:  Number of other nearest neighbor nodes around a node
            O:  Output dimension of the predicted properties

        Returns:
            properties mindspore.Tensor[float], [B,A,O]: prediction for the properties of the molecules

        """

        node_mask = None
        nodes_number = None
        if node_types is None:
            if self.fixed_atoms:
                node_types = self.node_types
                node_mask = self.node_mask
                nodes_number = self.nodes_number
                if  self.full_connect:
                    neighbors = self.neighbors
                    neighbor_mask = None
            else:
                # raise ValueError('node_types is miss')
                return None
        else:
            node_mask = F.expand_dims(node_types,-1) > 0
            nodes_number = F.cast(node_types>0,ms.float32)
            nodes_number = self.molsum(nodes_number,-1)

        if self.use_bonds:
            if bonds is None:
                if self.fixed_bonds:
                    exones = self.ones((positions.shape[0],1,1),ms.int32)
                    bonds = exones * self.bonds
                    bond_mask = exones * self.bond_mask
                else:
                    # raise ValueError('bonds is miss')
                    return None    
            if bond_mask is None:
                bond_mask = (bonds > 0)

        if neighbors is None:
            if self.full_connect:
                neighbors,neighbor_mask=self.fc_neighbors(node_types)
                if self.cut_shape:
                    atypes = F.cast(node_types>0,positions.dtype)
                    anum = self.reducesum(atypes,-1)
                    nmax = self.reducemax(anum)
                    nmax = F.cast(nmax,ms.int32)
                    nmax0 = int(nmax.asnumpy())
                    nmax1 = nmax0 - 1

                    node_types = node_types[:,:nmax0]
                    positions = positions[:,:nmax0,:]
                    neighbors = neighbors[:,:nmax0,:nmax1]
                    neighbor_mask = neighbor_mask[:,:nmax0,:nmax1]
            else:
                # raise ValueError('neighbors is miss')
                return None
        
        if self.use_distances:
            r_ij = self.distances(positions,neighbors,neighbor_mask) * self.dis_scale
        else:
            r_ij = 1
            neighbor_mask = bond_mask

        x, xlist = self.model(r_ij,node_types,node_mask,neighbors,neighbor_mask,bonds,bond_mask)

        if atom_types is None:
            atom_types = node_types
            atoms_number = nodes_number
        else:
            atoms_number = F.cast(atom_types>0,ms.float32)
            atoms_number = self.molsum(atoms_number,-1)

        far_neighbors = None
        far_mask = None
        far_rij = None
        if self.calc_far:
            if self.full_connect:
                far_neighbors = neighbors
                far_mask = neighbor_mask
                far_rij = r_ij
            else:
                far_rij = self.dis_scale * \
                    self.distances(positions,far_neighbors,far_mask,None,None)
        
        if self.multi_readouts:
            ytuple = ()
            for i in range(self.num_readout):
                yi = self.readout[i](x,xlist,node_mask,nodes_number,far_rij,far_neighbors,far_mask,atom_types,atoms_number)
                if self.unit_energy is not None:
                    yi = yi * self.output_scale[i]
                ytuple = ytuple + (yi,)
            y = self.concat(ytuple)
        else:
            y = self.readout(x,xlist,node_mask,nodes_number,far_rij,far_neighbors,far_mask,atom_types,atoms_number)
            if self.unit_energy is not None:
                y = y * self.output_scale

        return y