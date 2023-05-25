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
Cybertron tutorial 08: Read hyperparameters for networ trained with force
"""

import sys
import numpy as np
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train import load_checkpoint

if __name__ == '__main__':

    sys.path.append('..')

    from mindsponge.data import read_yaml
    from cybertron import Cybertron
    from cybertron.train import MAE, RMSE
    from cybertron.train import MolWithEvalCell

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # Initializing the network using a configuration file (yaml)
    config_file = 'Tutorial_C07/configure.yaml'
    config = read_yaml(config_file)
    net = Cybertron(**config)

    ckpt_file = 'Tutorial_C07/cybertron-molct-best.ckpt'
    load_checkpoint(ckpt_file, net)

    test_file = sys.path[0] + '/dataset_ethanol_origin_testset_1024.npz'
    test_data = np.load(test_file)

    scale = test_data['scale']
    shift = test_data['shift']

    net.set_scaleshift(scale=scale, shift=shift)

    net.print_info()

    ds_test = ds.NumpySlicesDataset(
        {'coordinate': test_data['coordinate'],
         'energy': test_data['label'],
         'force': test_data['force'],
         }, shuffle=True)
    data_keys = ds_test.column_names
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)
    eval_network = MolWithEvalCell(data_keys, net, calc_force=True)
    eval_network.print_info()

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_rmse = 'ForcesRMSE'
    model = Model(net, eval_network=eval_network,
                  metrics={energy_mae: MAE(0), forces_mae: MAE(1), forces_rmse: RMSE(1)})

    print('Evaluation with unnormalized test dataset:')
    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k, value in eval_metrics.items():
        info += k
        info += ': '
        info += str(value)
        info += ', '
    print(info)
