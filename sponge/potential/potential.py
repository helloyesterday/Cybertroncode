import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ..units import Units, global_units
from ..functions import gather_vectors
from ..tools import GetVector

class PotentialCell(Cell):
    def __init__(self,
                 exclude_index: Tensor = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__()

        if units is None:
            if length_unit is None and energy_unit is None:
                self.units = global_units
            else:
                self.units = Units(length_unit, energy_unit)
        else:
            if not isinstance(units, Units):
                raise TypeError(
                    'The type of units must be "Unit" but get type: '+str(type(units)))
            self.units = units

        self.use_pbc = use_pbc
        self._exclude_index = self._check_exclude_index(exclude_index)

        self.get_vector = GetVector(use_pbc)
        self.gather_atoms = gather_vectors

        self.identity = ops.Identity()

    @property
    def exclude_index(self) -> Tensor:
        if self._exclude_index is None:
            return None
        return self.identity(self._exclude_index)

    def _check_exclude_index(self, exclude_index: Tensor):
        if exclude_index is None:
            return None
        exclude_index = Tensor(exclude_index, ms.int32)
        if exclude_index.ndim == 2:
            exclude_index = F.expand_dims(exclude_index, 0)
        if exclude_index.ndim != 3:
            raise ValueError('The rank of exclude_index must be 2 or 3 but got: '
                                + str(exclude_index.shape))
        # (B,A,Ex)
        return Parameter(exclude_index, name='exclude_index', requires_grad=False)

    def set_exclude_index(self, exclude_index: Tensor):
        self._exclude_index = self._check_exclude_index(exclude_index)
        return self

    @property
    def length_unit(self):
        return self.units.length_unit

    @property
    def energy_unit(self):
        return self.units.energy_unit

    def set_pbc(self, use_pbc: bool = None):
        self.use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinates: Tensor,
                  neighbour_vectors: Tensor,
                  neighbour_distances: Tensor,
                  neighbour_index: Tensor,
                  neighbour_mask: Tensor = None,
                  pbc_box: Tensor = None
                  ):

        raise NotImplementedError
