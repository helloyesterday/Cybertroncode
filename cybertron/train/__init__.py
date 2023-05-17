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
Train
"""

from .wrapper import MoleculeWrapper, WithAdversarialLossCell
from .train import MolWithLossCell
from .eval import MolWithEvalCell
from .loss import MAELoss, MSELoss, CrossEntropyLoss
from .lr import TransformerLR
from .metric import MaxError, Error, MAE, MSE, MNE, RMSE, Loss
from .callback import TrainMonitor


__all__ = [
    'MoleculeWrapper',
    'WithAdversarialLossCell',
    'MolWithLossCell',
    'MolWithEvalCell',
]
__all__.extend(loss.__all__)
__all__.extend(lr.__all__)
__all__.extend(metric.__all__)
__all__.extend(callback.__all__)
