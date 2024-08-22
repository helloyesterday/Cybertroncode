# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
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
Training forcefield model with Cybertron.
"""

import sys
import time
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

if __name__ == '__main__':

    sys.path.append('../..')
    data_dir = './data'

    import os
    os.environ['GLOG_v'] = '4'

    from cybertron import Cybertron
    from cybertron.model import MolCT, SchNet
    from cybertron.embedding import MolEmbedding
    from cybertron.readout import AtomwiseReadout
    from cybertron.train import MolWithLossCell, MolWithEvalCell
    from cybertron.train.lr import TransformerLR
    from cybertron.train.loss import MSELoss
    from cybertron.train.metric import MAE, RMSE, Loss
    from cybertron.train.callback import TrainMonitor

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Set the backend.", default="GPU")
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.e)

    sys_name = data_dir + '/dataset/data_normed_'

    train_file = sys_name + 'trainset_83197_64_64.npz'
    valid_file = sys_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    atom_type = Tensor(train_data['atom_type'], ms.int32)
    scale = train_data['scale']
    shift = train_data['shift']

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

    net = Cybertron(embedding=emb, model=mod, readout=readout, atom_type=atom_type, length_unit='nm')
    net.set_scaleshift(scale=scale, shift=shift)

    conf_dir = data_dir + '/conf'
    net.save_configure('configure_MolCT' + '.yaml' , conf_dir)

    tot_params = 0
    for i, param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    net.print_info()

    N_EPOCH = 50
    REPEAT_TIME = 1
    BATCH_SIZE = 32

    ds_train = ds.NumpySlicesDataset(
        {'coordinate': train_data['coordinate'],
         'energy': train_data['label'],
         'force': train_data['force'],
         }, shuffle=True)

    data_keys = ds_train.column_names
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.repeat(REPEAT_TIME)

    force_dis = train_data['avg_force_dis']
    loss_network = MolWithLossCell(data_keys=data_keys,
                                   network=net,
                                   loss_fn=[MSELoss(), MSELoss(force_dis=force_dis)],
                                   calc_force=True,
                                   loss_weights=[1, 100],
                                   )
    loss_network.print_info()

    ds_valid = ds.NumpySlicesDataset(
        {'coordinate': valid_data['coordinate'],
         'energy': valid_data['label'],
         'force': valid_data['force'],
         }, shuffle=True)
    data_keys = ds_valid.column_names
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    eval_network = MolWithEvalCell(data_keys=data_keys,
                                   network=net,
                                   loss_fn=[MSELoss(), MSELoss(force_dis=force_dis)],
                                   calc_force=True,
                                   loss_weights=[1, 100],
                                   normed_evaldata=True
                                   )
    eval_network.print_info()

    lr = TransformerLR(learning_rate=1., warmup_steps=4000, dimension=dim_feature)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_rmse = 'ForcesRMSE'
    eval_loss = 'EvalLoss'
    model = Model(loss_network, eval_network=eval_network, optimizer=optim,
                  metrics={eval_loss: Loss(), energy_mae: MAE(0), forces_mae: MAE(1),
                           forces_rmse: RMSE(1)})

    ckpt_name = 'cybertron-' + net.model_name.lower()
    ckpt_dir = data_dir + '/ckpt'
    record_cb = TrainMonitor(model, ckpt_name, per_epoch=1, avg_steps=32,
                             directory=ckpt_dir, eval_dataset=ds_valid, best_ckpt_metrics=forces_rmse)

    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=ckpt_name, directory=ckpt_dir, config=config_ck)

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
