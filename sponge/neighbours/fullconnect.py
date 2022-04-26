
import mindspore as ms
from mindspore import numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

class FullConnectNeighbours(Cell):
    def __init__(self,num_atoms):
        super().__init__()
        # tot_atoms: A
        # tot_neigh: N =  A - 1
        self.num_atoms = num_atoms
        self.num_neigh = num_atoms -1
        
        # neighbours for no connection (A*N)
        # self.nnc = msnp.broadcast_to(msnp.arange(tot_atoms).shape(-1,1),(tot_atoms,tot_neigh))
        # (A,1)
        no_idx = msnp.arange(self.num_atoms).reshape(-1,1)

        # (N)
        nrange = msnp.arange(self.num_neigh)

        # neighbours for full connection (A,N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        fc_idx = nrange + F.cast(no_idx <= nrange,ms.int32)
        no_idx = msnp.broadcast_to(no_idx,(self.num_atoms,self.num_neigh))
        idx_mask = fc_idx > no_idx

        # (1,A,N)
        self.fc_idx = F.expand_dims(fc_idx,0)
        self.no_idx = F.expand_dims(no_idx,0)
        self.idx_mask = F.expand_dims(idx_mask,0)

        self.shape = (self.num_atoms,self.num_neigh)
        self.fc_mask = msnp.broadcast_to(Tensor(True),(1,)+self.shape)

        self.reduce_all = ops.ReduceAll()

    def get_full_neighbours(self):
        return self.fc_idx
        
    def construct(self, atom_mask: Tensor=None, exclude_index: Tensor=None):
        r"""Calculate the full connected neighbour list.

        Args:
            atom_mask (Tensor[bool] with shape (B,A), optional): Mask for atoms
            exclude_index (Tensor[int] with shape (B,A,E), optional): excluded index for each atom

        Returns:
            neighbours (Tensor[int] with shape (B,A,N)
            mask (Tensor[bool] with shape (B,A,N)

        """
        if atom_mask is None:
            neighbours = self.fc_idx
            mask = self.fc_mask
        else:
            # (B,A,N)
            nshape = (atom_mask.shape[0],) + self.shape
            
            # (B,1,N)
            mask0 = F.expand_dims(atom_mask[:,:-1],-2)
            mask1 = F.expand_dims(atom_mask[:,1:],-2)

            # (B,A,N)
            mask0 = msnp.broadcast_to(mask0,nshape)
            mask1 = msnp.broadcast_to(mask1,nshape)

            idx_mask = msnp.broadcast_to(self.idx_mask,nshape)
            mask = F.select(idx_mask,mask1,mask0)
            mask  = F.logical_and(F.expand_dims(atom_mask,-1),mask)

            fc_idx = msnp.broadcast_to(self.fc_idx,nshape)
            no_idx = msnp.broadcast_to(self.no_idx,nshape)

            neighbours = F.select(mask, fc_idx, no_idx)

        if exclude_index is not None:
            # (B,A,N,E) <- (B,A,N,1) vs (B,A,1,E)
            exc_mask = F.expand_dims(neighbours,-1) != F.expand_dims(exclude_index,-2)
            # (B,A,N)
            exc_mask = self.reduce_all(exc_mask,-1)
            mask = F.logical_and(mask,exc_mask)

        return neighbours,mask