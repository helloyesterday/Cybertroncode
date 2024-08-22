# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Read configure file
"""

import os
from typing import Union

path = os.getenv('MINDSPONGE_HOME')
if path:
    import sys
    sys.path.insert(0, path)
from sponge.data import read_yaml


def get_configure(configure: Union[str, dict], key: str = None) -> dict:
    """ Get template for molecule or residue.

    Args:

        configure (Union[dict, str):

    Returns:

        template (dict):  Template for molecule or residue

    """
    if configure is None:
        return configure

    if isinstance(configure, str):
        if os.path.exists(configure):
            filename = configure
        else:
            directory, _ = os.path.split(os.path.realpath(__file__))
            filename = os.path.join(directory, configure)
            if not os.path.exists(filename):
                raise ValueError(f'Cannot find configure file: {configure}')
        configure: dict = read_yaml(filename)

    if not isinstance(configure, dict):
        raise TypeError(f'The type of configure must be str or dict but got: {type(configure)}')

    if key is not None:
        if key in configure.keys():
            return configure.get(key)
        raise KeyError(f'Cannot find key "{key}" in configure.')

    return configure
