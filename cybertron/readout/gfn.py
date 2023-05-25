from typing import Tuple, Union
from mindspore import Tensor
from mindspore.nn import Cell, LayerNorm, CellList
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .readout import Readout
from ..layer import MLP
from ..cutoff import get_cutoff

from mindsponge.function import functions as func
from mindsponge.function import Length, Units, GLOBAL_UNITS

class GFNLayer(Cell):
    def __init__(self, 
                 dim_node_rep: int = None, 
                 dim_edge_rep: int = None, 
                 node_activation: Union[Cell,str] = None, 
                 edge_activation: Union[Cell,str] = None, 
                 ):
        super().__init__()
        self.node_update = MLP(dim_node_rep+dim_edge_rep,dim_node_rep,[dim_node_rep],activation=node_activation)
        self.edge_encoder = MLP(dim_edge_rep,dim_edge_rep,[dim_edge_rep],activation=edge_activation)
        self.edge_decoder = MLP(dim_edge_rep,1,[dim_edge_rep],activation=edge_activation)
        self.node_layernorm = LayerNorm([dim_node_rep])
        self.edge_layernorm = LayerNorm([dim_edge_rep])
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat(axis=-1)

    def construct(self, 
                  node_rep, 
                  edge_rep,
                  edge_msg, 
                  edge_dec, 
                  atom_mask, 
                  neighbours, 
                  neighbour_mask,
                  **kwargs):
        
        # update atom
        node_msg = self.reduce_sum(edge_msg,-2)
        node_edge = self.concat((node_rep,node_msg))
        node_rep += self.node_layernorm(self.node_update(node_edge)*self.expand_dims(atom_mask, -1))
                
        # update edge
        edge_rep += self.edge_layernorm(self.expand_dims(node_rep, -2) + func.gather_vector(node_rep, neighbours))
                
        # update force
        edge_msg = self.edge_encoder(edge_rep) * self.expand_dims(neighbour_mask, -1)
        edge_dec += self.edge_decoder(edge_msg) * self.expand_dims(neighbour_mask, -1)

        return node_rep, edge_rep, edge_msg, edge_dec


class GFNReadout(Readout):
    def __init__(self, 
                 dim_node_rep: int = None, 
                 dim_edge_rep: int = None, 
                 node_activation: Union[Cell,str] = None, 
                 edge_activation: Union[Cell,str] = None, 
                 iterations: int = None,
                 cutoff: Union[Length, float] = Length(1.0, 'nm'),
                 cutoff_fn = None,
                 length_unit: str = 'nm',
                 ndim: int = 1,
                 shape: Tuple[int] = (1,),
                 shared_parms = True,
                 **kwargs):
        super().__init__(ndim=ndim,shape=shape)

        if length_unit is None:
            self.units = GLOBAL_UNITS
        else:
            self.units = Units(length_unit)
        self.length_unit = self.units.length_unit

        self.cutoff = self.get_length(cutoff)
        self.cutoff_fn = get_cutoff(cutoff_fn, self.cutoff)
        self.iterations = iterations
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.expand_dims = P.ExpandDims()

        if shared_parms:
            self.read_out = CellList([GFNLayer(dim_node_rep,dim_edge_rep,node_activation,edge_activation)]*iterations)
        else:
            self.read_out = CellList([GFNLayer(dim_node_rep,dim_edge_rep,node_activation,edge_activation) for i in range(iterations)]) 

    def get_length(self, length, unit=None):
        if isinstance(length, Length):
            if unit is None:
                unit = self.units
            return Tensor(length(unit))
        return Tensor(length)
    
    def _calc_cutoffs(self, r_ij=1, neighbour_mask=None, atom_mask=None, bond_mask=None):
        if self.cutoff_fn is None:
            return F.ones_like(r_ij), neighbour_mask
        return self.cutoff_fn(r_ij, neighbour_mask)
        
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
                  **kwargs):
        
        c_ij, _ = self._calc_cutoffs(neigh_dis, neigh_mask, atom_mask, None)
        edge_msg = edge_emb
        edge_dec = 0

        for i in range(self.iterations):
            node_rep, edge_rep, edge_msg, edge_dec = self.read_out[i](node_rep, edge_rep, edge_msg, edge_dec, atom_mask, neigh_list, neigh_mask)

        dxi = self.reduce_sum(self.expand_dims(c_ij,-1) * neigh_vec * edge_dec, 2)        

        return dxi
        
