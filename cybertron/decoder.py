# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
# 
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mindspore import nn
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .block import MLP,Dense
from .block import PreActResidual
from .block import SeqPreActResidual
from .block import PreActDense

__all__ = [
    "Decoder",
    "get_decoder",
    "SimpleDecoder",
    "ResidualOutputBlock",
    ]

_DECODER_BY_KEY = dict()

def _decoder_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _DECODER_BY_KEY:
            _DECODER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _DECODER_BY_KEY:
                _DECODER_BY_KEY[alias] = cls
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

        self.reg_key = 'none'
        self.name = 'decoder'

        self.output = None
        self.activation = activation

    def construct(self, x):
        return self.output(x)

@_decoder_register('halve')
class HalveDecoder(Decoder):
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

        self.reg_key = 'halve'
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

        self.reg_key = 'residual'
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

_DECODER_BY_NAME = {decoder.__name__:decoder for decoder in _DECODER_BY_KEY.values()}

def get_decoder(
        decoder: str,
        n_in: int,
        n_out: int,
        activation: Cell=None,
        n_layers: int=1,
    ) -> Decoder:
    if decoder is None or isinstance(decoder,Decoder):
        return decoder

    if isinstance(decoder, str):
        if decoder.lower() == 'none':
            return None
        if decoder.lower() in _DECODER_BY_KEY.keys():
            return _DECODER_BY_KEY[decoder.lower()](
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                n_layers=n_layers,
            )
        elif decoder in _DECODER_BY_NAME.keys():
            return _DECODER_BY_NAME[decoder](
                n_in=n_in,
                n_out=n_out,
                activation=activation,
                n_layers=n_layers,
            )
        else:
            raise ValueError("The Decoder corresponding to '{}' was not found.".format(decoder))
    else:
        raise TypeError("Unsupported init type '{}'.".format(type(decoder)))