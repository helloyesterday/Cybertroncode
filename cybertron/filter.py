from mindspore.nn import Cell

from sponge.functions import get_integer

from .block import MLP, Dense, Residual

class DenseFilter(Cell):
    def __init__(
        self,
        num_basis,
        dim_filter,
        activation,
        n_hidden=1,
    ):
        super().__init__()

        self.num_basis = get_integer(num_basis)
        self.dim_filter = get_integer(dim_filter)

        if n_hidden > 0:
            hidden_layers = [self.dim_filter for _ in range(n_hidden)]
            self.dense_layers = MLP(self.num_basis,self.dim_filter,hidden_layers,activation=activation)
        else:
            self.dense_layers = Dense(self.num_basis,self.dim_filter,activation=activation)

    def construct(self,x):
        return self.dense_layers(x)

class ResFilter(Cell):
    def __init__(
        self,
        num_basis,
        dim_filter,
        activation,
        n_hidden=1,
    ):
        super().__init__()

        self.num_basis = get_integer(num_basis)
        self.dim_filter = get_integer(dim_filter)

        self.linear = Dense(self.num_basis,self.dim_filter,activation=None)
        self.residual = Residual(self.dim_filter,activation=activation,n_hidden=n_hidden)

    def construct(self,x):
        lx = self.linear(x)
        return self.residual(lx)