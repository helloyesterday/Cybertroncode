import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore import numpy as msnp

import sponge.functions as func

class GetVector(Cell):
    def __init__(self,use_pbc=None):
        super().__init__()

        self.get_vector = self.get_vector_default

        self.pbc = use_pbc
        self.set_pbc(use_pbc)

    def get_vector_without_pbc(self,position0,position1,pbc_box=None):
        return func.get_vector_without_pbc(position0,position1)

    def get_vector_with_pbc(self,position0,position1,pbc_box):
        return func.get_vector_with_pbc(position0,position1,pbc_box)

    def get_vector_default(self,position0,position1,pbc_box=None):
        return func.get_vector(position0,position1,pbc_box)

    def set_pbc(self,pbc=None):
        self.pbc = pbc
        if pbc is None:
            self.get_vector = self.get_vector_default
        else:
            if pbc:
                self.get_vector = self.get_vector_with_pbc
            else:
                self.get_vector = self.get_vector_without_pbc
        return self

    def construct(self, position0, position1, pbc_box=None):
        return self.get_vector(position0,position1,pbc_box)

class GetDistance(Cell):
    def __init__(self,use_pbc=None):
        super().__init__()

        self.get_vector = GetVector(use_pbc)

    def set_pbc(self,pbc):
        self.get_vector.set_pbc(pbc)
        return self

    def construct(self, position0, position1, pbc_box=None):
        vec = self.get_vector(position0,position1,pbc_box)
        return func.norm_m1(vec)

