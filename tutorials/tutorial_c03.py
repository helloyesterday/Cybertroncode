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
Cybertron tutorial 03: Use normalized dataset and validation dataset
"""

import sys
import time
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from mindsponge.data import write_yaml

if __name__ == '__main__':

    sys.path.append('..')

    from cybertron import Cybertron
    from cybertron.model import MolCT
    from cybertron.embedding import MolEmbedding
    from cybertron.readout import AtomwiseReadout
    from cybertron.train import TrainMonitor, MAELoss, MAE, Loss
    from cybertron.train import MolWithLossCell, MolWithEvalCell

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    data_name = sys.path[0] + '/dataset_qm9_normed_'
    train_file = data_name + 'trainset_1024.npz'
    valid_file = data_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    idx = [7]  # U0

    num_atom = int(train_data['num_atoms'])
    _scale = train_data['scale'][idx]
    shift = train_data['shift'][idx]
    value = train_data['type_ref'][:, idx]

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
        fixed_cycles=False,
        activation=activation,
        max_cycles=10,
        coupled_interaction=True,
    )

    readout = AtomwiseReadout(dim_output=1,
                              dim_node_rep=dim_feature,
                              activation=activation,
                              )
    
    net = Cybertron(embedding=emb, model=mod, readout=readout, num_atoms=num_atom)

    outdir = 'Tutorial_C03'
    write_yaml(net._kwargs, 'configure.yaml', outdir)

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
    data_key = ds_train.column_names
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.repeat(REPEAT_TIME)
    
    ds_valid = ds.NumpySlicesDataset(
        {'coordinate': valid_data['coordinate'],
         'atom_type': valid_data['atom_type'],
         'label': valid_data['label'][:, idx]}, shuffle=True)
    column_names = ds_valid.column_names
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    loss_network = MolWithLossCell(data_key, net, MAELoss())
    eval_network = MolWithEvalCell(data_key, net, MAELoss())

    # lr = 1e-3
    lr = nn.ExponentialDecayLR(
        learning_rate=1e-3, decay_rate=0.96, decay_steps=4, is_stair=True)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    eval_mae = 'EvalMAE'
    atom_mae = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(loss_network, optimizer=optim, eval_network=eval_network, metrics={
        eval_mae: MAE(), atom_mae: MAE(by_atoms=True), eval_loss: Loss()})

    outdir = 'Tutorial_C03'
    outname = outdir + '_' + net.model_name

    record_cb = TrainMonitor(model, outname, per_step=16, avg_steps=16,
                             directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=eval_loss)

    config_ck = CheckpointConfig(
        save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(
        prefix=outname, directory=outdir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    model.train(N_EPOCH, ds_train, callbacks=[
                record_cb, ckpoint_cb], dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print("Training Fininshed!")
    print("Training Time: %02d:%02d:%02d" % (h, m, s))
