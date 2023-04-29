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

Cybertron tutorial 03: Load & test network

Key points:
    1) Initialize the network using a configuration file (yaml).
    2) Load parameters for the network using a checkpoint files (ckpt).
    3) Set scale and shift for network.
    4) Use original or normalized dataset for test.

"""

import sys
import numpy as np
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model

if __name__ == '__main__':

    sys.path.append('..')

    from mindsponge.data import read_yaml, load_checkpoint

    from cybertron import Cybertron
    from cybertron.train import MAE, Loss
    from cybertron.train import MolWithEvalCell, MAELoss

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    # Initializing the network using a configuration file (yaml)
    config_file = 'Tutorial_C02/configure.yaml'
    config = read_yaml(config_file)
    net = Cybertron(**config)

    # Loading parameters for the network using a checkpoint files (ckpt)
    ckpt_file = 'Tutorial_C02/cybertron-molct-8_32.ckpt'
    load_checkpoint(ckpt_file, net)

    test_file = sys.path[0] + '/dataset_qm9_origin_testset_1024.npz'
    test_data = np.load(test_file)

    idx = [7]  # U0

    num_atom = test_data['num_atoms']
    scale = test_data['scale'][idx]
    shift = test_data['shift'][idx]
    type_ref = test_data['type_ref'][:, idx]

    # Setting scale and shift
    net.set_scaleshift(scale=scale, shift=shift, type_ref=type_ref)

    tot_params = 0
    for i, param in enumerate(net.get_parameters()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    net.print_info()

    # Use original (unnormalized) data for evaluation
    ds_test = ds.NumpySlicesDataset(
        {'coordinate': test_data['coordinate'],
         'atom_type': test_data['atom_type'],
         'label': test_data['label'][:, idx]}, shuffle=False)
    data_keys = ds_test.column_names
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)

    # NOTE: When using unnormalized data for evaluation,
    # the argument `normed_evaldata` should be set to `False`
    # Default: False
    eval_network = MolWithEvalCell(data_keys, net, MAELoss())

    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(net, eval_network=eval_network, metrics=
                  {eval_mae: MAE(), atom_mae: MAE(by_atoms=True), eval_loss: Loss()})

    print('Evaluation with unnormalized test dataset:')
    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k, value in eval_metrics.items():
        info += k
        info += ': '
        info += str(value)
        info += ', '
    print(info)

    normed_test_file = sys.path[0] + '/dataset_qm9_normed_testset_1024.npz'
    normed_test_file = np.load(normed_test_file)

    # Using normaed dataset
    ds_test_normed = ds.NumpySlicesDataset(
        {'coordinate': normed_test_file['coordinate'],
         'atom_type': normed_test_file['atom_type'],
         'label': normed_test_file['label'][:, idx]}, shuffle=False)
    data_keys = ds_test_normed.column_names
    ds_test_normed = ds_test_normed.batch(1024)
    ds_test_normed = ds_test_normed.repeat(1)

    # NOTE: When using normalized data for evaluation,
    # the argument `normed_evaldata` should be set to `True`
    # Default: False
    eval_network0 = MolWithEvalCell(data_keys, net, MAELoss(), normed_evaldata=True)

    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    eval_loss = 'Evalloss'
    model0 = Model(net, eval_network=eval_network0, metrics=
                   {eval_mae: MAE(), atom_mae: MAE(by_atoms=True), eval_loss: Loss()})

    print('Evaluation with normalized test dataset:')
    eval_metrics = model0.eval(ds_test_normed, dataset_sink_mode=False)
    info = ''
    for k, value in eval_metrics.items():
        info += k
        info += ': '
        info += str(value)
        info += ', '
    print(info)
