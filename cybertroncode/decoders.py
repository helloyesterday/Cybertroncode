from os import X_OK
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from cybertroncode.units import units
from cybertroncode.blocks import MLP,Dense
from cybertroncode.blocks import PreActResidual
from cybertroncode.blocks import SeqPreActResidual
from cybertroncode.blocks import PreActDense

__all__ = [
    "Decoder",
    "get_decoder",
    "SimpleDecoder",
    "ResidualOutputBlock",
]

_DECODER_ALIAS = dict()

def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_ALIAS:
            _DECODER_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _DECODER_ALIAS:
                _DECODER_ALIAS[alias] = cls
        return cls
    return alias_reg

class Decoder(nn.Cell):
    def __init__(self,
        n_in,
        n_out=1,
        activation=None,
        n_layers=1,
    ):
        super().__init__()

        self.name = 'decoder'

        self.output = None
        self.activation = activation

    def construct(self, x):
        return self.output(x)

@_decoder_register('halve')
class SimpleDecoder(Decoder):
    def __init__(self,
        n_in,
        n_out,
        activation,
        n_layers=1,
    ):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.name = 'halve'

        if n_layers > 0:
            n_hiddens = []
            dim = n_in
            for i in range(n_layers):
                dim = dim // 2
                if dim < n_out:
                    raise ValueError("The dimension of hidden layer is smaller than output dimension")
                n_hiddens.append(dim)
            self.output = MLP(n_in, n_out, n_hiddens, activation=activation)
        else:
            self.output = Dense(n_in, n_out, activation=activation)

    def __str__(self):
        return 'halve'

@_decoder_register('residual')
class ResidualOutputBlock(Decoder):
    def __init__(
        self,
        n_in,
        n_out,
        activation,
        n_layers=1,
    ):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )

        self.name = 'residual'

        if n_layers == 1:
            output_residual = PreActResidual(n_in,activation=activation)
        else:
            output_residual = SeqPreActResidual(n_in,activation=activation,n_res=n_layers)

        self.output = nn.SequentialCell([
            output_residual,
            PreActDense(n_in,n_out,activation=activation),
        ])

    def __str__(self):
        return 'residual'

def get_decoder(
        obj,
        n_in,
        n_out,
        activation=None,
        n_layers=1,
    ):
    if obj is None or isinstance(obj,Decoder):
        return obj
    elif isinstance(obj, str):
        if obj not in _DECODER_ALIAS.keys():
            raise ValueError("The class corresponding to '{}' was not found.".format(obj))
        return _DECODER_ALIAS[obj.lower()](
            n_in=n_in,
            n_out=n_out,
            activation=activation,
            n_layers=n_layers,
        )
    else:
        raise TypeError("Unsupported init type '{}'.".format(type(obj)))