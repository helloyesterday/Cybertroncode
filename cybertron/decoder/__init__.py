# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
"""
Deep molecular model
"""

from typing import Union
from mindspore.nn import Cell

from .decoder import Decoder, _DECODER_BY_KEY
from .halve import HalveDecoder
from .residual import ResidualOutputBlock

__all__ = [
    'Decoder',
    'HalveDecoder',
    'ResidualOutputBlock',
    'get_decoder',
]


_DECODER_BY_NAME = {
    decoder.__name__: decoder for decoder in _DECODER_BY_KEY.values()}


def get_decoder(cls_name: str,
                dim_in: int,
                dim_out: int,
                activation: Union[Cell, str] = None,
                n_layers: int = 1,
                **kwargs
                ) -> Decoder:
    """get decoder by name"""
    if cls_name is None or isinstance(cls_name, Decoder):
        return cls_name

    if isinstance(cls_name, dict):
        return get_decoder(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None
        if cls_name.lower() in _DECODER_BY_KEY.keys():
            return _DECODER_BY_KEY[cls_name.lower()](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                **kwargs
            )
        if cls_name in _DECODER_BY_NAME.keys():
            return _DECODER_BY_NAME[cls_name](
                dim_in=dim_in,
                dim_out=dim_out,
                activation=activation,
                n_layers=n_layers,
                **kwargs
            )

        raise ValueError(
            "The Decoder corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported init type '{}'.".format(type(cls_name)))
