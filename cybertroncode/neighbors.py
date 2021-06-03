import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from cybertroncode.units import units

__all__ = [
    "GatherNeighbors",
    "Distances",
]

class GatherNeighbors(nn.Cell):
    def __init__(self,dim,fixed_neigh=False):
        super().__init__()
        self.fixed_neigh = fixed_neigh

        self.broad_ones = P.Ones()((1,1,dim),ms.int32)

        self.gatherd = P.GatherD()

    def construct(self,inputs,neighbors):
        # Construct auxiliary index vector
        ns = neighbors.shape
        
        # Get atomic positions of all neighboring indices

        if self.fixed_neigh:
            return F.gather(inputs,neighbors[0],-2)
        else:
            # [B, A, N] -> [B, A*N, 1]
            neigh_idx = F.reshape(neighbors,(ns[0],ns[1]*ns[2],-1))
            # [B, A*N, V] = [B, A*N, V] * [1, 1, V]
            neigh_idx = neigh_idx * self.broad_ones
            # [B, A*N, V] gather from [B, A, V]
            outputs = self.gatherd(inputs,1,neigh_idx)
            # [B, A, N, V]
            return F.reshape(outputs,(ns[0],ns[1],ns[2],-1))

class Distances(nn.Cell):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        neighbors_fixed (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self,fixed_atoms=False,dim=3):
        super().__init__()
        self.fixed_atoms=fixed_atoms
        self.reducesum = P.ReduceSum()
        self.pow = P.Pow()
        self.gatherd = P.GatherD()
        self.norm = nn.Norm(-1)

        self.gather_neighbors = GatherNeighbors(dim,fixed_atoms)

    def construct(
        self, positions, neighbors, neighbor_mask=None, cell=None, cell_offsets=None
        ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (ms.Tensor[float]): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (ms.Tensor[int]): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) or (N_at x N_nbh) shape.
            cell (ms.tensor[float], optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (ms.Tensor[float], optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (ms.Tensor[bool], optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            ms.Tensor[float]: layer output of (N_b x N_at x N_nbh) shape.

        """

        pos_xyz = self.gather_neighbors(positions,neighbors)

        # Subtract positions of central atoms to get distance vectors
        dist_vec = pos_xyz - F.expand_dims(positions,-2)

        distances = self.norm(dist_vec)
        
        if neighbor_mask is not None:
            r_large = F.ones_like(distances)*5e4
            distances = F.select(neighbor_mask,distances,r_large)

        return distances