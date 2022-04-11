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

import sponge.functions as func
from sponge.units import Units
from sponge.tools import GetVector
from sponge.units import global_units

class Colvar(Cell):
    r"""Base class for collective variables.

        The function "construct" of Colvar must has the argument "coordinates"

    Args:
        dim_output (int):   The output dimension, i.e., the last dimension of output Tensor.
        periodic (bool):    Whether the CV is periodic or not
        use_pbc (bool):     Whether to calculate the CV at periodic boundary condition (PBC).
                            If this term is "True", the CV will ALWAYS be calculated at PBC 
                            and the input term "pbc_box" must be provided.
                            If this term is "False", the CV will NEVER be calulcated at PBC, 
                            even if the input term "pbc_box" is provided.
                            Default value is "None", which means whether the CV will be 
                            calculted at PBC will be determined by the input "pbc_box".

   """
    def __init__(self,
        dim_output,
        periodic=False,
        period=None,
        use_pbc=None,
        unit_length=None,
    ):
        super().__init__()

        self.dim_output = dim_output

        self.get_vector = GetVector(use_pbc)
        self.pbc = use_pbc

        if unit_length is not None:
            self.use_global_units = False
            self.units = Units(unit_length)
        else:
            self.use_global_units = True
            self.units = global_units

        # the CV is periodic or not
        if isinstance(periodic,bool):
            periodic = Tensor([periodic,]*self.dim_output,ms.bool_)
        elif isinstance(periodic,(list,tuple)):
            if len(periodic) != self.dim_output:
                if len(periodic) == 1:
                    periodic = Tensor(periodic*self.dim_output,ms.bool_)
                else:
                    raise ValueError("The number of periodic mismatch")
        else:
            raise TypeError("Unspported type for periodic:"+str(type(periodic)))

        self.periodic = F.reshape(periodic,(1,1,self.dim_output))

        self.any_periodic = self.periodic.any()
        self.all_periodic = self.periodic.all()

    def length_unit(self):
        return self.units.length_unit()

    def difference_in_pbc(self,vec,pbc_box):
        return func.difference_in_pbc(vec,pbc_box)

    def calc_displacement(self,vec,pbc_box):
        return func.calc_displacement(vec,pbc_box)

    def set_pbc(self,pbc):
        self.pbc = pbc
        self.get_vector.set_pbc(pbc)

    def construct(self, coordinates, pbc_box=None):
        raise NotImplementedError