import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import nn
from mindspore import ms_function
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore.ops import constexpr
from mindspore import numpy as msnp

PI = 3.141592653589793238462643383279502884197169399375105820974944592307

inv = ops.Inv()
keepdim_sum = ops.ReduceSum(keep_dims=True)
keepdim_mean = ops.ReduceMean(keep_dims=True)
keep_norm_last_dim = nn.Norm(axis=-1,keep_dims=True)
norm_last_dim = nn.Norm(axis=-1,keep_dims=False)
reduce_any = ops.ReduceAny()
reduce_all = ops.ReduceAll()
concat_last_dim = ops.Concat(-1)
concat_penulti = ops.Concat(-2)

@ms_function
def pbc_box_reshape(pbc_box: Tensor, ndim: int) -> Tensor:
    r"""Reshape the pbc_box as the same ndim.

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        pbc_box (ms.Tensor[float]): (B,D) periodic box
        ndim (int): the rank (ndim) of the pbc_box

    Returns:
        pbc_box (ms.Tensor[float]): (B,1,..,1,D) reshaped pbc_box with rank of ndim

    """
    if ndim <= 2:
        return pbc_box
    else:
        shape = pbc_box.shape[:1] + (1,) *(ndim -2) + pbc_box.shape[-1:]
        return F.reshape(pbc_box,shape)

@ms_function
def calc_number_of_displace_box(vector: Tensor, pbc_box: Tensor, shift: float=0) -> Tensor:
    r"""Calculate the number of box to displace a vector at PBC box.

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        vector (ms.Tensor[float]): (B,...,D) position A
        pbc_box (ms.Tensor[float]): (B,D) periodic box
        shift (float, optional): shift of pbc box

    Returns:
        nbox (ms.Tensor[float]): (B,...,D) number of box to displace a vector at PBC box.

    """
    box = pbc_box_reshape(pbc_box,vector.ndim)
    return F.floor(vector/box-shift)

@ms_function
def calc_displacement(vector: Tensor, pbc_box: Tensor, shift: float=0) -> Tensor:
    r"""Calculate the displacement to make a vector at PBC box.

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        vector (ms.Tensor[float]): (B,...,D) position A
        pbc_box (ms.Tensor[float]): (B,D) periodic box
        shift (float, optional): shift of pbc box

    Returns:
        displacement (ms.Tensor[float]): (B,...,D) displacement to make a vector at PBC box

    """
    nbox = calc_number_of_displace_box(vector,pbc_box,shift)
    return pbc_box * nbox

@ms_function
def position_in_pbc(position: Tensor, pbc_box: Tensor, shift: float=0) -> Tensor:
    r"""Align the position at a box at perodic bundary condition

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position (ms.Tensor[float]): (B,...,D) position A
        pbc_box (ms.Tensor[float]): (B,D) periodic box
        shift (float, optional): shift of pbc box

    Returns:
        position (ms.Tensor[float]): (B,...,D) vector from position_a to position_b

    """
    displacement = calc_displacement(position,pbc_box,shift)
    return position - displacement

@ms_function
def difference_in_pbc(difference: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Make the difference of vecters at the range from -0.5 box to 0.5 box
        at perodic bundary condition. (-0.5box < difference < 0.5box)

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        difference (ms.Tensor[float]): (B,...,D) position A
        pbc_box (ms.Tensor[float]): (B,D) periodic box

    Returns:
        difference (ms.Tensor[float]): (B,...,D) vector from position_a to position_b

    """

    return position_in_pbc(difference,pbc_box,-0.5)

@ms_function
def get_vector_without_pbc(position_a: Tensor, position_b: Tensor) -> Tensor:
    r"""Compute vector from position A to position B without perodic bundary condition.

        B (int): Batchsize, i.e. number of walkers in simulation
        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        pbc_box (ms.Tensor[float]): (B,D) periodic box

    Returns:
        vector (ms.Tensor[float]): (B,...,D) vector from position_a to position_b

    """
    return position_b - position_a

@ms_function
def get_vector_with_pbc(position_a: Tensor, position_b: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute vector from position A to position B at perodic bundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        pbc_box (ms.Tensor[float]): (B,D) periodic box

    Returns:
        vector (ms.Tensor[float]): (B,...,D) vector from position_a to position_b

    """
    vec_ab = get_vector_without_pbc(position_a,position_b)
    return difference_in_pbc(vec_ab,pbc_box)

@ms_function
def get_vector(position_a: Tensor, position_b: Tensor, pbc_box: Tensor=None) -> Tensor:
    r"""Compute vector from position A to position B.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        pbc_box (ms.Tensor[float], optional): (B,D) periodic box

    Returns:
        vector (ms.Tensor[float]): (B,...,1) vector from position_a to position_b

    """
    vec_ab = get_vector_without_pbc(position_a,position_b)
    if pbc_box is None:
        return vec_ab
    else:
        return difference_in_pbc(vec_ab,pbc_box)


@ms_function
def gather_vectors(vectors: Tensor, index: Tensor) -> Tensor:
    r"""Gather vectors of index atoms.

    Args:
        vectors (ms.Tensor[float]): (B,A,D)
        index (ms.Tensor[int]): (B,...)

    Returns:
        neighbour_atoms (ms.Tensor[float]): (B,...,D)

    """

    if index.shape[0] == 1:
        return F.gather(vectors,index[0],-2)
    elif vectors.shape[0] == 1:
        return F.gather(vectors[0],index,-2)
    else:
        # (B,N,M)
        origin_shape = index.shape
        # (B,N*M,1) <- (B,N,M)
        index = F.reshape(index,(origin_shape[0],-1,1))

        # (B,N*M,D) <- (B,N*M) + (D,)
        broad_shape = index.shape[:-1] + vectors.shape[-1:]
        # (B,N*M,D) <- (B,N*M,1)
        index = msnp.broadcast_to(index,broad_shape)

        # (B,N*M,D) <- (B,N,D)
        neigh_atoms = F.gather_d(vectors,-2,index)
        # (B,N,M,D) <- (B,N,M) + (D,)
        output_shape = origin_shape + vectors.shape[-1:]

        # (B,N,M,D)
        return F.reshape(neigh_atoms,output_shape)

@ms_function
def gather_values(values: Tensor, index: Tensor) -> Tensor:
    r"""Get values of index atoms.

    Args:
        values (ms.Tensor[float]): (B,X)
        index (ms.Tensor[int]): (B,...)

    Returns:
        neighbour_atoms (ms.Tensor[float]): (B,...)

    """

    if index.shape[0] == 1:
        return F.gather(values,index[0],-1)
    elif values.shape[0] == 1:
        return F.gather(values[0],index,-1)
    else:
        # (B,N,M)
        origin_shape = index.shape
        # (B,N*M) <- (B,N,M)
        index = F.reshape(index,(origin_shape[0],-1))

        # (B,N*M) <- (B,X)
        neigh_values = F.gather_d(values,-1,index)

        # (B,N,M)
        return F.reshape(neigh_values,origin_shape)

@ms_function
def calc_distance_without_pbc(position_a: Tensor,position_b: Tensor) -> Tensor:
    r"""Compute distance between position A and B without perodic bundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B

    Returns:
        distance (ms.Tensor[float]): (B,...,1) distance between position_a and position_b

    """
    vec = get_vector_without_pbc(position_a,position_b)
    return keep_norm_last_dim(vec)

@ms_function
def calc_distance_with_pbc(position_a: Tensor, position_b: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute distance between position A and B at perodic bundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        pbc_box (ms.Tensor[float]): (B,D) periodic box

    Returns:
        distance (ms.Tensor[float]): (B,...,1) distance between position_a and position_b

    """
    vec = get_vector_with_pbc(position_a,position_b,pbc_box)
    return keep_norm_last_dim(vec)

@ms_function
def calc_distance(position_a: Tensor, position_b: Tensor, pbc_box: Tensor=None) -> Tensor:
    r"""Compute distance between position A and B

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        pbc_box (ms.Tensor[float], optional): (B,D) periodic box

    Returns:
        distance (ms.Tensor[float]): (B,...,1) distance between position_a and position_b

    """
    vec = get_vector_without_pbc(position_a,position_b)
    if pbc_box is not None:
        vec = difference_in_pbc(vec,pbc_box)
    return keep_norm_last_dim(vec)

@ms_function
def calc_angle_between_vectors(vector1: Tensor, vector2: Tensor) -> Tensor:
    r"""Compute angle between two vectors.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        vector1 (ms.Tensor[float]): (B,...,D) the 1st vector
        vector2 (ms.Tensor[float]): (B,...,D) the 2nd vector

    Returns:
        angle (ms.Tensor[float]): (B,...,1) angle between vector1 and vector2

    """

    # [X,1] <- [X,3]
    dis1 = keep_norm_last_dim(vector1)
    dis2 = keep_norm_last_dim(vector2)
    # [X,1] <- [X,3]
    dot12 = keepdim_sum(vector1 * vector2, -1)
    # [X,1]/[X,3]
    cos_theta = dot12 / dis1 / dis2
    return F.acos(cos_theta)

@ms_function
def calc_angle_without_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor) -> Tensor:
    r"""Compute angle formed by three positions A-B-C without periodic boundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C

    Returns:
        angle (ms.Tensor[float]): (B,...,1) angle of ABC

    """
    # (...,D)
    vec_ba = get_vector_without_pbc(position_b,position_a)
    vec_bc = get_vector_without_pbc(position_b,position_c)
    return calc_angle_between_vectors(vec_ba,vec_bc)

@ms_function
def calc_angle_with_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute angle formed by three positions A-B-C at periodic boundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C
        pbc_box (ms.Tensor[float]): (B,D) periodic box with shape

    Returns:
        angle (ms.Tensor[float]): (B,...,1) angle of ABC

    """
    # (...,D)
    vec_ba = get_vector_with_pbc(position_b,position_a,pbc_box)
    vec_bc = get_vector_with_pbc(position_b,position_c,pbc_box)
    return calc_angle_between_vectors(vec_ba,vec_bc)

@ms_function
def calc_angle(position_a,position_b: Tensor, position_c: Tensor, pbc_box: Tensor=None) -> Tensor:
    r"""Compute angle formed by three positions A-B-C.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C
        pbc_box (ms.Tensor[float], optional): (B,D) periodic box with shape

    Returns:
        angle (ms.Tensor[float]): (B,...,1) angle of ABC

    """
    # (...,D)
    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b,position_a)
        vec_bc = get_vector_without_pbc(position_b,position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b,position_a,pbc_box)
        vec_bc = get_vector_with_pbc(position_b,position_c,pbc_box)
    return calc_angle_between_vectors(vec_ba,vec_bc)

@ms_function
def calc_torsion_for_vectors(vector1: Tensor, vector2: Tensor, vector3: Tensor) -> Tensor:
    r"""Compute torsion angle formed by three vectors.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        vector1 (ms.Tensor[float]): (B,...,D) the 1st vector
        vector2 (ms.Tensor[float]): (B,...,D) the 2nd vector
        vector3 (ms.Tensor[float]): (B,...,D) the 3rd vector

    Returns:
        torsion (ms.Tensor[float]): (B,...,1) torsion angle formed by three vector1, vector2 and vector3.

    """
    # (B,...,D) <- (B,...,1)
    v2norm = keep_norm_last_dim(vector2)
    # (B,...,D) = (B,...,D) / (...,1)
    norm_vec2 = vector2 / v2norm

    # (B,...,D)
    vec_a = msnp.cross(norm_vec2,vector1)
    vec_b = msnp.cross(vector3,norm_vec2)
    cross_ab = msnp.cross(vec_a,vec_b)

    # (B,...,1)
    sin_phi = keepdim_sum(cross_ab*norm_vec2,-1)
    cos_phi = keepdim_sum(vec_a*vec_b,-1)

    return F.atan2(-sin_phi,cos_phi)

@ms_function
def calc_torsion_without_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, position_d: Tensor) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D without periodic boundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C
        position_d (ms.Tensor[float]): (B,...,D) position D

    Returns:
        torsion (ms.Tensor[float]): (B,...,1) torsion angle ABCD

    """
    vec_ba = get_vector_without_pbc(position_b,position_a)
    vec_cb = get_vector_without_pbc(position_c,position_b)
    vec_dc = get_vector_without_pbc(position_d,position_c)
    return calc_torsion_for_vectors(vec_ba,vec_cb,vec_dc)

@ms_function
def calc_torsion_with_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, position_d: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D at periodic boundary condition.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C
        position_d (ms.Tensor[float]): (B,...,D) position D
        pbc_box (ms.Tensor[float]): (B,D) periodic box with shape

    Returns:
        torsion (ms.Tensor[float]): (B,...,1) torsion angle ABCD

    """

    vec_ba = get_vector_with_pbc(position_b,position_a,pbc_box)
    vec_cb = get_vector_with_pbc(position_c,position_b,pbc_box)
    vec_dc = get_vector_with_pbc(position_d,position_c,pbc_box)
    return calc_torsion_for_vectors(vec_ba,vec_cb,vec_dc)

@ms_function
def calc_torsion(position_a: Tensor, position_b: Tensor, position_c: Tensor, position_d: Tensor, pbc_box: Tensor=None) -> Tensor:
    r"""Compute torsion angle formed by four positions A-B-C-D.

        D (int): Dimension of the simulation system. Usually is 3.

    Args:
        position_a (ms.Tensor[float]): (B,...,D) position A
        position_b (ms.Tensor[float]): (B,...,D) position B
        position_c (ms.Tensor[float]): (B,...,D) position C
        position_d (ms.Tensor[float]): (B,...,D) position D
        pbc_box (ms.Tensor[float], optional): (B,D) periodic box with shape

    Returns:
        torsion (ms.Tensor[float]): (B,...,1) torsion angle ABCD

    """

    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b,position_a)
        vec_cb = get_vector_without_pbc(position_c,position_b)
        vec_dc = get_vector_without_pbc(position_d,position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b,position_a,pbc_box)
        vec_cb = get_vector_with_pbc(position_c,position_b,pbc_box)
        vec_dc = get_vector_with_pbc(position_d,position_c,pbc_box)

    return calc_torsion_for_vectors(vec_ba,vec_cb,vec_dc)

def get_full_connect_index(num_atoms: int, exclude_index: Tensor=None) -> Tensor:
    num_neigh = num_atoms - 1
    arange = F.expand_dims(msnp.arange(num_atoms,dtype=ms.int32),-1)
    nrange = F.expand_dims(msnp.arange(num_neigh,dtype=ms.int32),0)
    
    amat = msnp.broadcast_to(arange,(num_atoms,num_neigh))
    nmat = msnp.broadcast_to(nrange,(num_atoms,num_neigh))

    # neighbours for full connection (A*N)
    # [[1,2,3,...,N],
    #  [0,2,3,...,N],
    #  [0,1,3,....N],
    #  .............,
    #  [0,1,2,...,N-1]]
    index = nmat + F.cast(amat <= nmat, ms.int32)
    index = F.expand_dims(index,0)

    mask = None
    if exclude_index is not None:
        mask = get_exculde_mask(index,exclude_index)
        amat = F.expand_dims(amat,0)
        index = F.select(mask,index,amat)

    return index,mask

def get_exculde_mask(index: Tensor,exclude_index: Tensor) -> Tensor:
    # (B,A,N) -> (B,A,N,1)
    exindex = F.expand_dims(index,-1)
    # (1,A,1,E) <- (1,A,E)
    exclude_index = F.expand_dims(exclude_index,-2)

    # (B,A,N) <- (B,A,N,E) <- (B,A,N,1) == (1,A,1,E)
    mask = reduce_all(exindex != exclude_index,-1)

    return mask

@ms_function
def get_kinetic_energy(m: Tensor,v: Tensor) -> Tensor:
    # (B,A) <- (B,A,D)
    v2 = F.reduce_sum(v*v,-1)
    # (B,A) <- (1,A) * (B,A)
    k = 0.5 * m * v2
    return keepdim_sum(k,-1)

def get_integer(value: Tensor) -> int:
    if value is None:
        return None
    if isinstance(value,Tensor):
        value = value.asnumpy()
    return int(value)

if __name__ == "__main__":

    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")