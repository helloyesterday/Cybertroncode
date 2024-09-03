from typing import Tuple, Union
from mindspore import Tensor
from mindspore.nn import Cell, LayerNorm, CellList, Dense
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .readout import Readout, _readout_register
from ..layer import MLP

import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.function import functions as func
from sponge.function import Length, Units, GLOBAL_UNITS

class GFNLayer(Cell):
    def __init__(self, 
                 dim_node_rep: int = None, 
                 dim_edge_rep: int = None, 
                 node_activation: Union[Cell,str] = None, 
                 edge_activation: Union[Cell,str] = None,
                 ):
        super().__init__()
        # self.node_update = MLP(dim_node_rep+dim_edge_rep,dim_node_rep,[dim_node_rep],activation=node_activation)
        # self.edge_encoder = MLP(dim_edge_rep,dim_edge_rep,[dim_edge_rep],activation=edge_activation)
        self.edge_decoder = MLP(dim_edge_rep,1,[dim_edge_rep],activation=edge_activation)
        self.node_update = Dense(dim_node_rep+dim_edge_rep,dim_node_rep,activation=node_activation)
        self.edge_encoder = Dense(dim_edge_rep,dim_edge_rep,activation=edge_activation)
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
                  neighbour_mask,
                  **kwargs):

        # update atom
        node_msg = self.reduce_sum(edge_msg,-2)
        node_edge = self.concat((node_rep,node_msg))
        node_layernorm = self.node_layernorm(self.node_update(node_edge)*self.expand_dims(atom_mask, -1))
        node_rep += node_layernorm
                
        # update edge
        edge_layernorm = self.edge_layernorm(self.expand_dims(node_rep, -2) + self.expand_dims(node_rep, 1))
        edge_rep += edge_layernorm
                
        # update force
        edge_msg = self.edge_encoder(edge_rep) * self.expand_dims(neighbour_mask, -1)
        edge_dec += self.edge_decoder(edge_msg) * self.expand_dims(neighbour_mask, -1)

        return node_rep, edge_rep, edge_msg, edge_dec


@_readout_register('gfn')
class GFNReadout(Readout):
    def __init__(self, 
                 dim_node_rep: int = None, 
                 dim_edge_rep: int = None, 
                 node_activation: Union[Cell,str] = None, 
                 edge_activation: Union[Cell,str] = None, 
                 iterations: int = None,
                 shared_parms = True,
                 **kwargs):
        super().__init__()

        self._ndim = 2
        self._shape = (1,)

        self.iterations = iterations
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.expand_dims = P.ExpandDims()
        self.shared_parms = shared_parms

        if shared_parms:
            self.read_out = CellList([GFNLayer(dim_node_rep,dim_edge_rep,node_activation,edge_activation)]*iterations)
        else:
            self.read_out = CellList([GFNLayer(dim_node_rep,dim_edge_rep,node_activation,edge_activation) for i in range(iterations)])

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f" Activation function: {self.activation}")
        print(ret+gap+f" Representation dimension: {self.dim_node_rep}")
        print(ret+gap+f" Readout iterations: {self.iterations}")
        print(ret+gap+f" Shape of readout: {self.shape}")
        print(ret+gap+f" Rank (ndim) of readout: {self.ndim}")
        print(ret+gap+f" Whether used shared parameters: {self.shared_parms}")
        print('-'*80)
        return self 
        
    def construct(self, 
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_cutoff: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  distance: Tensor = None,
                  dis_mask: Tensor = None,
                  dis_vec: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs):
        
        edge_msg = edge_emb
        edge_dec = 0

        for i in range(self.iterations):
            node_rep, edge_rep, edge_msg, edge_dec = self.read_out[i](node_rep, edge_rep, edge_msg, edge_dec, atom_mask, dis_mask)

        dxi = self.reduce_sum(self.expand_dims(edge_cutoff,-1) * dis_vec * edge_dec, 2)   

        return dxi[:,0].expand_dims(1)
        
