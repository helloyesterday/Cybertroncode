# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
# 
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from pyparsing import Char

from sponge.functions import get_integer
from sponge.hyperparam import set_class_into_hyper_param,set_hyper_parameter
from sponge.hyperparam import get_hyper_string,get_hyper_parameter,get_class_parameters
from sponge.units import Units,global_units,Length

from .block import Residual,Dense
from .interaction import SchNetInteraction
from .interaction import PhysNetModule
from .interaction import NeuralInteractionUnit
from .base import GraphNorm
from .filter import ResFilter,DenseFilter
from .cutoff import Cutoff,get_cutoff
from .rbf import get_rbf
from .activation import get_activation

__all__ = [
    "MolecularModel",
    "SchNet",
    "PhysNet",
    "MolCT",
    ]

class MolecularModel(Cell):
    r"""Basic class for graph neural network (GNN) based deep molecular model

    Args:
        num_atom_types (int): maximum number of atomic types
        num_basis (int): number of the serial of radical basis functions (RBF)
        dim_feature (int): dimension of the vectors for atomic embedding
        atom_types (ms.Tensor[int], optional): atomic index 
        rbf(nn.Cell, optional): the algorithm to calculate RBF
        cutoff_fn (nn.Cell, optional): the algorithm to calculate cutoff.

    """

    def __init__(
        self,
        dim_feature: int=128,
        n_interaction: int=3,
        activation: Cell=None,
        cutoff: Length=Length(1,'nm'),
        cutoff_fn: Cutoff=None,
        rbf: Cell=None,
        r_self: Length=None,
        coupled_interaction: bool=False,
        use_distance: bool=True,
        use_bond: bool=False,
        use_graph_norm: bool=False,
        public_dis_filter: bool=False,
        public_bond_filter: bool=False,
        num_atom_types: int=64,
        num_bond_types: int=16,
        length_unit: bool='nm',
        hyper_param: dict=None,
    ):
        super().__init__()

        self.network_name='MolecularModel'

        if hyper_param is not None:
            num_atom_types = get_hyper_parameter(hyper_param,'num_atom_types')
            num_bond_types = get_hyper_parameter(hyper_param,'num_bond_types')
            dim_feature = get_hyper_parameter(hyper_param,'dim_feature')
            n_interaction = get_hyper_parameter(hyper_param,'n_interaction')
            activation = get_class_parameters(hyper_param,'activation')
            cutoff = get_hyper_parameter(hyper_param,'cutoff')
            cutoff_fn = get_class_parameters(hyper_param,'cutoff_fn')
            rbf = get_class_parameters(hyper_param,'rbf')
            r_self = get_hyper_parameter(hyper_param,'r_self')
            coupled_interaction = get_hyper_parameter(hyper_param,'coupled_interaction')
            use_distance = get_hyper_parameter(hyper_param,'use_distance')
            use_bond = get_hyper_parameter(hyper_param,'use_bond')
            public_dis_filter = get_hyper_parameter(hyper_param,'public_dis_filter')
            public_bond_filter = get_hyper_parameter(hyper_param,'public_bond_filter')
            use_graph_norm = get_hyper_parameter(hyper_param,'use_graph_norm')
            length_unit = get_hyper_string(hyper_param,'length_unit')

        if length_unit is None:
            self.units = global_units
        else:
            self.units = Units(length_unit)
        self.length_unit = self.units.length_unit()

        self.num_atom_types = get_integer(num_atom_types)
        self.num_bond_types = get_integer(num_bond_types)
        self.dim_feature = get_integer(dim_feature)
        self.n_interaction = get_integer(n_interaction)
        self.r_self = r_self
        self.coupled_interaction = Tensor(coupled_interaction,ms.bool_)
        self.use_distance = self.broadcast_to_interactions(use_distance,'use_distance')
        self.use_bond = self.broadcast_to_interactions(use_bond,'use_bond')
        self.public_dis_filter = Tensor(public_dis_filter,ms.bool_)
        self.public_bond_filter = Tensor(public_bond_filter,ms.bool_)
        self.use_graph_norm = Tensor(use_graph_norm,ms.bool_)

        self.activation = get_activation(activation)

        self.cutoff = None
        self.cutoff_fn = None
        self.rbf = None
        if self.use_distance.any():
            self.cutoff = self.get_length(cutoff)
            self.cutoff_fn = get_cutoff(cutoff_fn,self.cutoff)
            self.rbf = get_rbf(rbf,self.cutoff,length_unit=self.length_unit)
            
        self.r_self_ex = None
        if self.r_self is not None:
            self.r_self = self.get_length(self.r_self)
            self.r_self_ex = F.expand_dims(self.r_self,0)

        self.atom_embedding = nn.Embedding(self.num_atom_types, self.dim_feature, use_one_hot=True, embedding_table=Normal(1.0))
        self.bond_embedding = None

        self.num_basis = self.rbf.num_basis

        self.interactions = None
        self.interaction_typenames = []

        self.calc_distance = self.use_distance.any()
        self.calc_bond = self.use_bond.any()

        self.use_pub_norm = False
        
        if self.use_graph_norm:
            if self.use_pub_norm:
                self.graph_norm = nn.CellList(
                    [ GraphNorm(dim_feature) * self.n_interaction ]
                )
            else:
                self.graph_norm = nn.CellList(
                    [ GraphNorm(dim_feature) for _ in range(self.n_interaction) ]
                )
        else:
            self.graph_norm = None

        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.concat = P.Concat(-1)

        self.hyper_param = dict()
        self.hyper_types = {
            'num_atom_types'      : 'int',
            'num_bond_types'      : 'int',
            'dim_feature'         : 'int',
            'n_interaction'       : 'int',
            'activation'          : 'Cell',
            'cutoff'              : 'float',
            'cutoff_fn'           : 'Cell',
            'rbf'                 : 'Cell',
            'r_self'              : 'float',
            'coupled_interaction' : 'bool',
            'use_distance'        : 'bool',
            'use_bond'            : 'bool',
            'public_dis_filter'   : 'bool',
            'public_bond_filter'  : 'bool',
            'use_graph_norm'      : 'bool',
            'length_unit'         : 'str',
        }

    def set_hyper_param(self):
        set_hyper_parameter(self.hyper_param,'name',self.cls_name)
        set_class_into_hyper_param(self.hyper_param,self.hyper_types,self)
        return self

    def get_length(self,length,unit=None):
        if isinstance(length,Length):
            if unit is None:
                unit = self.units
            return Tensor(length(unit),ms.float32)
        else:
            return Tensor(length,ms.float32)

    def broadcast_to_interactions(self,value,name:str):
        tensor = Tensor(value)
        size = tensor.size
        if self.coupled_interaction:
            if size > 1:
                raise ValueError('The size of "'+name+'" must be 1 when "coupled_interaction" is "True"')
        else:
            if size != self.n_interaction:
                if size != 1:
                    raise ValueError('"The size of "'+name+'" ('+str(size)+
                        ') must be equal to "n_interaction" ('+str(self.n_interaction)+')!')
                tensor = msnp.broadcast_to(tensor,(self.n_interaction,))
        return tensor
                

    def print_info(self, num_retraction: int=3, num_gap: int=3, char: str='-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+' Deep molecular model: ',self.network_name)
        print('-'*80)
        print(ret+gap+' Length unit: ' + self.units.length_unit_name())
        print(ret+gap+' Atom embedding size: ' + str(self.num_atom_types))
        print(ret+gap+' Cutoff distance: ' + str(self.cutoff) + ' ' + self.length_unit)
        print(ret+gap+' Radical basis function (RBF): ' + str(self.rbf.cls_name))
        self.rbf.print_info(num_retraction=num_retraction+num_gap,num_gap=num_gap,char=char)
        print(ret+gap+' Calculate distance: ' + ('Yes' if self.calc_distance else 'No'))
        print(ret+gap+' Calculate bond: ' + ('Yes' if self.calc_bond else 'No'))
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print('-'*80)
        if self.coupled_interaction:
            print(ret+gap+' Using coupled interaction with '+str(self.n_interaction)+' layers:')
            print('-'*80)
            print(ret+gap+gap+' '+self.interactions[0].name)
            self.interactions[0].print_info(num_retraction=num_retraction+num_gap,num_gap=num_gap,char=char)
        else:
            print(ret+gap+' Using '+str(self.n_interaction)+' independent interaction layers:')
            print('-'*80)
            for i,inter in enumerate(self.interactions):
                print(ret+gap+' '+str(i)+'. '+inter.name)
                inter.print_info(num_retraction=num_retraction+num_gap,num_gap=num_gap,char=char)

    def _get_self_interaction(self,atom_mask):
        # (B,A,1)
        r_ii = msnp.full_like(atom_mask,self.r_self)
        r_large = msnp.full_like(r_ii,5e4)
        r_ii = F.select(atom_mask,r_ii,r_large)
        c_ii = F.ones_like(r_ii) * atom_mask
        return r_ii,c_ii

    def _calc_cutoffs(self,r_ij=1,neighbour_mask=None,atom_mask=None,bond_mask=None):
        if self.calc_distance:
            if self.cutoff_fn is None:
                return F.ones_like(r_ij),neighbour_mask
            else:
                return self.cutoff_fn(r_ij,neighbour_mask)
        else:
            mask = None
            if bond_mask is not None:
                mask = self.concat((atom_mask,bond_mask))
            return F.cast(mask>0,ms.float32),mask

    def _get_self_cutoff(self,atom_mask):
        return F.cast(atom_mask,ms.float32)

    def _get_rbf(self,dis):
        if self.rbf is None:
            rbf = F.expand_dims(dis,-1)
        else:
            rbf = self.rbf(dis)

        return rbf

    def construct(
        self,
        r_ij: Tensor=1,
        atom_types: Tensor=None,
        atom_mask: Tensor=None,
        neighbours: Tensor=None,
        neighbour_mask: Tensor=None,
        bonds: Tensor=None,
        bond_mask: Tensor=None,
    ):
        """Compute interaction output.

        Args:
            r_ij (ms.Tensor[float], [B, A, N]): distances between atoms.
            atom_types (ms.Tensor[int], optional): atomic number
            atom_mask (ms.Tensor[int], optional): mask of atomic number
            neighbours (ms.Tensor[int], [B, A, N], optional): neighbour indices.
            neighbour_mask (ms.Tensor[bool], optional): mask of neighbour indices.

        Returns:
            representation: (ms.Tensor[float], [B, A, F]) representation of atoms.

        """

        bsize = r_ij.shape[0] if self.calc_distance else bonds.shape[0]

        e =  self.atom_embedding(atom_types)
        if atom_types.shape[0] != bsize:
            e = msnp.broadcast_to(e,(bsize,)+e.shape[1:])
            atom_mask = msnp.broadcast_to(atom_mask,(bsize,)+atom_mask.shape[1:])
        
        if self.calc_distance:
            nbatch = r_ij.shape[0]
            natoms = r_ij.shape[1]

            f_ij = self._get_rbf(r_ij)
            f_ii = 0 if self.r_self is None else self._get_rbf(self.r_self_ex)
        else:
            f_ii = 1
            f_ij = 1
            nbatch = bonds.shape[0]
            natoms = bonds.shape[1]

        if self.calc_bond:
            b_ii = self.zeros((nbatch,natoms),ms.int32)
            b_ii = self.bond_embedding(b_ii)
            
            b_ij = self.bond_embedding(bonds)

            if bond_mask is not None:
                b_ij = b_ij * F.expand_dims(bond_mask,-1)
        else:
            b_ii = 0
            b_ij = 0

        # apply cutoff
        c_ij, mask = self._calc_cutoffs(r_ij,neighbour_mask,atom_mask,bond_mask)
        c_ii = None if self.r_self is None else self._get_self_cutoff(atom_mask)

        # continuous-filter convolution interaction block followed by Dense layer
        x = e
        n_interaction = len(self.interactions)
        xlist = []
        for i in range(n_interaction):
            if self.r_self is None:
                x = self.interactions[i](x, f_ij, b_ij, c_ij, neighbours, mask)
            else:
                x = self.interactions[i](x, f_ij, b_ij, c_ij, neighbours, mask, e, f_ii, b_ii, c_ii, atom_mask)
            if self.use_graph_norm:
                x = self.graph_norm[i](x)
            xlist.append(x)
        return x,xlist
        
class SchNet(MolecularModel):
    r"""SchNet Model.
    
    References:
        Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; Müller K.-R.,
        SchNet - a deep learning architecture for molceules and materials.
        The Journal of Chemical Physics 148 (24), 241722. 2018.

    Args:

    """
    def __init__(
        self,
        dim_feature: int=64,
        dim_filter: int=64,
        n_interaction: int=3,
        activation: Cell='ssp',
        cutoff: float=Length(1,'nm'),
        cutoff_fn: Cell='cosine',
        rbf: Cell='gaussian',
        normalize_filter: bool=False,
        coupled_interaction: bool=False,
        use_graph_norm: bool=False,
        public_dis_filter: bool=False,
        num_atom_types: int=64,
        length_unit: str='nm',
        hyper_param: dict=None,
    ):
        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff = cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=None,
            coupled_interaction=coupled_interaction,
            use_distance=True,
            use_bond=False,
            use_graph_norm=use_graph_norm,
            public_dis_filter=public_dis_filter,
            num_atom_types=num_atom_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
            )
        self.reg_key = 'schnet'
        self.network_name = 'SchNet'

        if hyper_param is not None:
            dim_filter = get_hyper_parameter(hyper_param,'dim_filter')
            normalize_filter = get_hyper_parameter(hyper_param,'normalize_filter')

        if self.calc_bond:
            raise ValueError('SchNet cannot supported bond information!')

        self.dim_filter = self.broadcast_to_interactions(dim_filter,'dim_filter')
        self.normalize_filter = self.broadcast_to_interactions(normalize_filter,'normalize_filter')

        self.set_hyper_param()

        self.filter = None
        if self.public_dis_filter and (not self.coupled_interaction):
            self.filter = DenseFilter(self.num_basis,self.dim_filter,self.activation)
        # block for computing interaction
        if self.coupled_interaction:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        dis_filter=DenseFilter(self.num_basis,self.dim_filter,self.activation),
                        normalize_filter=self.normalize_filter,
                    )
                ]
                * self.n_interaction
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_filter=self.dim_filter[i],
                        activation=self.activation,
                        dis_filter=self.filter if self.public_dis_filter
                                               else DenseFilter(self.num_basis,self.dim_filter[i],self.activation),
                        normalize_filter=self.normalize_filter[i],
                    )
                    for i in range(self.n_interaction)
                ]
            )

    def set_hyper_param(self):
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param,'dim_filter',self.dim_filter)
        set_hyper_parameter(self.hyper_param,'normalize_filter',self.normalize_filter)
        return self

class PhysNet(MolecularModel):
    r"""PhysNet Model
    
    References:
        Unke, O. T. and Meuwly, M.,
        PhysNet: A neural network for predicting energyies, forces, dipole moments, and partial charges.
        The Journal of Chemical Theory and Computation 2019, 15(6), 3678-3693.

    Args:

    """
    def __init__(
        self,
        dim_feature: int=128,
        n_interaction: int=5,
        activation: Cell='ssp',
        cutoff: float=Length(1,'nm'),
        cutoff_fn: Cell='smooth',
        rbf: Cell='log_gaussian',
        public_dis_filter: bool=False,
        use_graph_norm: bool=False,
        coupled_interaction: bool=False,
        num_atom_types: int=64,
        n_inter_residual: int=3,
        n_outer_residual: int=2,
        length_unit: str='nm',
        hyper_param: dict=None,
    ):
        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff = cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=None,
            coupled_interaction=coupled_interaction,
            use_distance=True,
            use_bond=False,
            use_graph_norm=use_graph_norm,
            public_dis_filter=public_dis_filter,
            num_atom_types=num_atom_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
            )

        self.reg_key = 'physnet'
        self.network_name = 'PhysNet'

        if hyper_param is not None:
            n_inter_residual = get_hyper_parameter(hyper_param,'n_inter_residual')
            n_outer_residual = get_hyper_parameter(hyper_param,'n_outer_residual')

        self.n_inter_residual = get_integer(n_inter_residual)
        self.n_outer_residual = get_integer(n_outer_residual)

        self.set_hyper_param()

        self.filter = None
        if self.public_dis_filter and (not self.coupled_interaction):
            self.filter = Dense(self.num_basis,self.dim_feature,has_bias=False,activation=None),

        # block for computing interaction
        if self.coupled_interaction:
            self.interaction_typenames = ['D0',] * self.n_interaction
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        dis_filter=Dense(self.num_basis,self.dim_feature,has_bias=False,activation=None),
                        dim_feature=self.dim_feature,
                        activation=self.activation,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction_typenames = [ 'D' + str(i) for i in range(self.n_interaction) ]
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.CellList(
                [
                    PhysNetModule(
                        dis_filter=self.filter if self.public_dis_filter
                                else Dense(self.num_basis,self.dim_feature,has_bias=False,activation=None),
                        dim_feature=self.dim_feature,
                        activation=self.activation,
                        n_inter_residual=self.n_inter_residual,
                        n_outer_residual=self.n_outer_residual,
                    )
                    for _ in range(self.n_interaction)
                ]
            )
        
        self.readout = None

    def set_hyper_param(self):
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param,'n_inter_residual',self.n_inter_residual)
        set_hyper_parameter(self.hyper_param,'n_outer_residual',self.n_outer_residual)
        return self
        
class MolCT(MolecularModel):
    r"""Molecular Configuration Transformer (MolCT) Model
    
    References:
        Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
        Molecular CT: unifying geometry and representation learning for molecules at different scales
        ArXiv: 2012.11816

    Args:
        

    """
    def __init__(
        self,
        dim_feature: int=128,
        n_interaction: int=3,
        n_heads: int=8,
        max_cycles: int=10,
        activation: Cell='swish',
        cutoff: Length=Length(1,'nm'),
        cutoff_fn: Cell='smooth',
        rbf: Cell='log_gaussian',
        r_self: Length=Length(0.05,'nm'),
        use_distance: bool=True,
        use_bond: bool=False,
        public_dis_filter: bool=True,
        public_bond_filter: bool=True,
        num_atom_types: int=64,
        num_bond_types: int=16,
        use_feed_forward: bool=False,
        fixed_cycles: bool=False,
        coupled_interaction: bool=False,
        length_unit: str='nm',
        act_threshold=0.9,
        hyper_param: dict=None,
    ):
        super().__init__(
            dim_feature=dim_feature,
            n_interaction=n_interaction,
            activation=activation,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf=rbf,
            r_self=r_self,
            coupled_interaction=coupled_interaction,
            use_distance=use_distance,
            use_bond=use_bond,
            public_dis_filter=public_dis_filter,
            public_bond_filter=public_bond_filter,
            use_graph_norm=False,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            length_unit=length_unit,
            hyper_param=hyper_param,
            )

        self.reg_key = 'molct'
        self.network_name = 'MolCT'

        if hyper_param is not None:
            n_heads = get_hyper_parameter(hyper_param,'n_heads')
            max_cycles = get_hyper_parameter(hyper_param,'max_cycles')
            use_feed_forward = get_hyper_parameter(hyper_param,'use_feed_forward')
            fixed_cycles = get_hyper_parameter(hyper_param,'fixed_cycles')
            act_threshold = get_hyper_parameter(hyper_param,'act_threshold')

        if self.r_self is None:
            raise ValueError('"r_self" cannot be "None" at MolCT.')
        self.self_dis_tensor = F.expand_dims(self.r_self,0)

        self.n_heads = self.broadcast_to_interactions(n_heads,'n_heads')
        self.max_cycles = self.broadcast_to_interactions(max_cycles,'max_cycles')
        self.use_feed_forward = self.broadcast_to_interactions(use_feed_forward,'use_feed_forward')
        self.fixed_cycles = self.broadcast_to_interactions(fixed_cycles,'fixed_cycles')
        self.act_threshold = self.broadcast_to_interactions(act_threshold,'act_threshold')

        self.set_hyper_param()

        self.dis_filter = None
        if self.calc_distance and self.public_dis_filter and (not self.coupled_interaction):
            self.dis_filter = ResFilter(self.num_basis,self.dim_feature,self.activation)
        
        self.bond_embedding = None
        self.bond_filter = None
        if self.calc_bond:
            self.bond_embedding = nn.Embedding(self.num_bond_types, self.dim_feature, use_one_hot=True, embedding_table=Normal(1.0))
            if self.calc_bond and self.public_bond_filter and (not self.coupled_interaction):
                self.bond_filter = Residual(self.dim_feature,activation=self.activation)

        if self.coupled_interaction:
            self.interactions = nn.CellList(
                [
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        dis_filter=(ResFilter(self.num_basis,self.dim_feature,self.activation) if self.use_distance else None),
                        bond_filter=(Residual(self.dim_feature,activation=self.activation) if self.use_bond else None),
                        use_feed_forward=self.use_feed_forward,
                        fixed_cycles=self.fixed_cycles,
                        act_threshold=self.act_threshold,
                    )
                ]
                * self.n_interaction
            )
        else:
            interaction_list = []
            for i in range(self.n_interaction):
                dis_filter = None
                if self.use_distance[i]:
                    if self.public_dis_filter:
                        dis_filter = self.dis_filter
                    else:
                        dis_filter = ResFilter(self.num_basis,self.dim_feature,self.activation)
                bond_filter = None
                if self.use_bond[i]:
                    if self.public_bond_filter:
                        bond_filter = self.bond_filter
                    else:
                        bond_filter = Residual(self.dim_feature,activation=self.activation)
                
                interaction_list.append(
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads[i],
                        max_cycles=self.max_cycles[i],
                        activation=self.activation,
                        dis_filter=dis_filter,
                        bond_filter=bond_filter,
                        use_feed_forward=self.use_feed_forward[i],
                        fixed_cycles=self.fixed_cycles[i],
                        act_threshold=self.act_threshold[i],
                    )
                )
            self.interactions = nn.CellList(interaction_list)

    def set_hyper_param(self):
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param,'n_heads',self.n_heads)
        set_hyper_parameter(self.hyper_param,'max_cycles',self.max_cycles)
        set_hyper_parameter(self.hyper_param,'use_feed_forward',self.use_feed_forward)
        set_hyper_parameter(self.hyper_param,'fixed_cycles',self.fixed_cycles)
        set_hyper_parameter(self.hyper_param,'act_threshold',self.act_threshold)
        return self

_MOLECULAR_MODEL_BY_KEY = {
    'molct':MolCT,
    'schnet':SchNet,
    'physnet':PhysNet
}

_MOLECULAR_MODEL_BY_NAME = {model.__name__:model for model in _MOLECULAR_MODEL_BY_KEY.values()}

def get_molecular_model(model,length_unit=None) -> MolecularModel:
    if isinstance(model,MolecularModel):
        return model
    if model is None:
        return None

    hyper_param = None
    if isinstance(model,dict):
        if 'name' not in model.keys():
            raise KeyError('Cannot find the key "name" in model dict!')
        hyper_param = model
        model = get_hyper_string(hyper_param,'name')

    if isinstance(model,str):
        if model.lower() == 'none':
            return None
        if model.lower() in _MOLECULAR_MODEL_BY_KEY.keys():
            return _MOLECULAR_MODEL_BY_KEY[model.lower()](
                length_unit=length_unit,
                hyper_param=hyper_param,
            )
        elif model in _MOLECULAR_MODEL_BY_NAME.keys():
            return _MOLECULAR_MODEL_BY_NAME[model](
                length_unit=length_unit,
                hyper_param=hyper_param,
            )
        else:
            raise ValueError("The MolecularModel corresponding to '{}' was not found.".format(model))
    else:
        raise TypeError("Unsupported MolecularModel type '{}'.".format(type(model)))