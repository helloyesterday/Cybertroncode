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
Cybertron tutorial 05: Multi-task training

Key points:
    1) Set multi readouts.
    2) Set dataset with multi labels.
    3) set metrics for multi labels.

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

    from cybertron import Cybertron
    from cybertron.model import MolCT
    from cybertron.embedding import MolEmbedding
    from cybertron.readout import AtomwiseReadout
    from cybertron.train import MAE, Loss, MAELoss
    from cybertron.train import MolWithLossCell, MolWithEvalCell
    from cybertron.train import TrainMonitor
    from cybertron.train import TransformerLR

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    data_name = sys.path[0] + '/dataset_qm9_normed_'
    train_file = data_name + 'trainset_1024.npz'
    valid_file = data_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    idx = [7, 8, 9, 10]  # U0, U, G, H

    num_atom = train_data['num_atoms']

    # Set multi scale, shift and type_ref
    scale = [train_data['scale'][[i]] for i in idx]
    shift = [train_data['shift'][[i]] for i in idx]
    type_ref = [train_data['type_ref'][:, [i]] for i in idx]

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

    # Set multi readouts
    readout0 = AtomwiseReadout(1, dim_feature, activation)
    readout1 = AtomwiseReadout(1, dim_feature, activation)
    readout2 = AtomwiseReadout(1, dim_feature, activation)
    readout3 = AtomwiseReadout(1, dim_feature, activation)

    net = Cybertron(embedding=emb,
                    model=mod,
                    readout=[readout0, readout1, readout2, readout3],
                    num_atoms=num_atom, length_unit='nm'
                    )

    net.set_scaleshift(scale=scale, shift=shift, type_ref=type_ref)

    tot_params = 0
    for i, param in enumerate(net.get_parameters()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)

    net.print_info()

    N_EPOCH = 8
    REPEAT_TIME = 1
    BATCH_SIZE = 32

    # set training dataset with multi labels
    ds_train = ds.NumpySlicesDataset(
        {'coordinate': train_data['coordinate'],
         'atom_type': train_data['atom_type'],
         'label0': train_data['label'][:, [7]],
         'label1': train_data['label'][:, [8]],
         'label2': train_data['label'][:, [9]],
         'label3': train_data['label'][:, [10]],
         }, shuffle=True)
    data_keys = ds_train.column_names
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
    ds_train = ds_train.repeat(REPEAT_TIME)

    loss_network = MolWithLossCell(data_keys, net, MAELoss())

    # set valiation dataset with multi labels
    ds_valid = ds.NumpySlicesDataset(
        {'coordinate': valid_data['coordinate'],
         'atom_type': valid_data['atom_type'],
         'label0': valid_data['label'][:, [7]],
         'label1': valid_data['label'][:, [8]],
         'label2': valid_data['label'][:, [9]],
         'label3': valid_data['label'][:, [10]],
         }, shuffle=True)
    data_keys = ds_valid.column_names
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    eval_network = MolWithEvalCell(data_keys, net, MAELoss(), normed_evaldata=True)

    lr = TransformerLR(learning_rate=1., warmup_steps=4000, dimension=dim_feature)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    # set metrics for multi labels.
    model = Model(loss_network,
                  optimizer=optim,
                  eval_network=eval_network,
                  metrics={'EvalLoss': Loss(),
                           'Label0MAE': MAE(0),
                           'Label1MAE': MAE(1),
                           'Label2MAE': MAE(2),
                           'Label3MAE': MAE(3),}
                  )

    outdir = 'Tutorial_C05'
    outname = outdir + '_' + net.model_name
    record_cb = TrainMonitor(model, outname, per_step=16, avg_steps=16,
                             directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics='Evalloss')

    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=outname, directory=outdir, config=config_ck)

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
