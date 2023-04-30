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
Cybertron tutorial 05: Validate and test the model.

Key points:
    1) Set `normed_evaldata` according to whether the validation dataset is normalized or not.
    2) Automatically save best models by setting `best_ckpt_metrics`

"""

import sys
import time
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

if __name__ == '__main__':

    sys.path.append('..')

    from mindsponge.data import read_yaml, load_checkpoint

    from cybertron import Cybertron
    from cybertron.model import MolCT
    from cybertron.embedding import MolEmbedding
    from cybertron.readout import AtomwiseReadout
    from cybertron.train import MolWithLossCell, MolWithEvalCell
    from cybertron.train.loss import MAELoss
    from cybertron.train.metric import MAE, Loss
    from cybertron.train.callback import TrainMonitor

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    train_file = sys.path[0] + '/dataset_qm9_normed_trainset_1024.npz'
    valid_file = sys.path[0] + '/dataset_qm9_origin_validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    idx = [7]  # U0

    num_atom = train_data['num_atoms']
    scale = train_data['scale'][idx]
    shift = train_data['shift'][idx]
    type_ref = train_data['type_ref'][:, idx]

    dim_feature = 128
    activation = 'silu'

    emb = MolEmbedding(
        dim_node=dim_feature,
        emb_dis=True,
        emb_bond=False,
        cutoff=1,
        cutoff_fn='smooth',
        rbf_fn='log_gaussian',
        activation=activation,
        length_unit='nm',
    )

    mod = MolCT(
        dim_feature=dim_feature,
        dim_edge_emb=emb.dim_edge,
        n_interaction=3,
        n_heads=8,
        activation=activation,
    )

    readout = AtomwiseReadout(
        dim_output=1,
        dim_node_rep=dim_feature,
        activation=activation,
    )

    net = Cybertron(embedding=emb, model=mod, readout=readout, num_atoms=num_atom)
    # Setting scale and shift (necessary for validation)
    net.set_scaleshift(scale=scale, shift=shift, type_ref=type_ref)

    outdir = 'Tutorial_C05'
    net.save_configure('configure.yaml', outdir)

    net.print_info()

    tot_params = 0
    for i, param in enumerate(net.get_parameters()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    N_EPOCH = 8
    REPEAT_TIME = 1
    BATCH_SIZE = 32

    ds_train = ds.NumpySlicesDataset(
        {'coordinate': train_data['coordinate'],
         'atom_type': train_data['atom_type'],
         'label': train_data['label'][:, idx]}, shuffle=True)
    data_keys = ds_train.column_names
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.repeat(REPEAT_TIME)

    loss_network = MolWithLossCell(data_keys, net, MAELoss())
    loss_network.print_info()

    ds_valid = ds.NumpySlicesDataset(
        {'coordinate': valid_data['coordinate'],
         'atom_type': valid_data['atom_type'],
         'label': valid_data['label'][:, idx]}, shuffle=True)
    data_keys = ds_valid.column_names
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    # NOTE: When using unnormalized data for evaluation,
    # the argument `normed_evaldata` should be set to `False`
    # Default: False
    eval_network = MolWithEvalCell(data_keys, net, MAELoss())
    eval_network.print_info()

    # lr = 1e-3
    lr = nn.ExponentialDecayLR(learning_rate=1e-3, decay_rate=0.96, decay_steps=4, is_stair=True)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    # Set metric functions
    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(loss_network, optimizer=optim, eval_network=eval_network, metrics={
        eval_mae: MAE(), atom_mae: MAE(by_atoms=True), eval_loss: Loss()})

    ckpt_name = 'cybertron-' + net.model_name.lower()

    # Automatically save the best model based on a specified metric (best_ckpt_metrics)
    record_cb = TrainMonitor(model, ckpt_name, per_step=32, avg_steps=32,
                             directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=eval_loss)

    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=ckpt_name, directory=outdir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    model.train(N_EPOCH, ds_train, callbacks=[record_cb, ckpoint_cb], dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print("Training Fininshed!")
    print("Training Time: %02d:%02d:%02d" % (h, m, s))

    # Initializing the network using a configuration file (yaml)
    config_file = 'Tutorial_C05/configure.yaml'
    config = read_yaml(config_file)
    net0 = Cybertron(**config)

    # Loading the best checkpoint file.
    ckpt_file = 'Tutorial_C05/cybertron-molct-best.ckpt'
    # NOTE: If the scale, shift and type_ref has already been set in the previous training,
    # the relevant parameters will be stored in the ckpt file.
    # So it is no need to set them again after loading the parameters.
    load_checkpoint(ckpt_file, net0)

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

    print('Test model:')
    eval_metrics = model0.eval(ds_test_normed, dataset_sink_mode=False)
    info = ''
    for k, value in eval_metrics.items():
        info += k
        info += ': '
        info += str(value)
        info += ', '
    print(info)
