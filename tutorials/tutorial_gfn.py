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
Cybertron tutorial GFN: Using GFN readout
"""

import time
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor,context
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from cybertron import Cybertron
from cybertron.model import MolCT
from cybertron.embedding import MolEmbedding
from cybertron.readout import GFNReadout
from cybertron.train import MolWithLossCell, MolWithEvalCell, TransformerLR
from cybertron.train.loss import MSELoss
from cybertron.train.metric import MAE, Loss, RMSE
from cybertron.train.callback import TrainMonitor

from mindsponge.function import Length

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=6)

    N_EPOCH = 128
    REPEAT_TIME = 1
    BATCH_SIZE = 32

    sys_name = 'dataset_ethanol_normed_'

    train_file = sys_name + 'trainset_1024.npz'
    valid_file = sys_name + 'validset_128.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    num_sample = train_data['coordinate'].shape[0]

    atom_type = np.array(train_data['atom_type'], np.int32).reshape(1,-1)
    pbc_box = np.ones((1,3),dtype=np.float32) * 99.9
    scale = train_data['scale']
    scale2 = np.sqrt(np.var(train_data['force']))

    ds_train = ds.NumpySlicesDataset(
        {'coordinate': train_data['coordinate'],
         'atom_type': atom_type.repeat(num_sample,0),
         'pbc_box': pbc_box.repeat(num_sample,0),
         'label': train_data['force']/scale2,
         }, shuffle=True)
    
    ds_valid = ds.NumpySlicesDataset(
        {'coordinate': valid_data['coordinate'],
         'atom_type': atom_type.repeat(128,0),
         'pbc_box': pbc_box.repeat(128,0),
         'label': valid_data['force']/scale2,
         }, shuffle=True)

    data_keys = ds_train.column_names
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.repeat(REPEAT_TIME)
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    dim_feature = 128
    activation = 'silu'

    emb = MolEmbedding(
        dim_node=dim_feature,
        emb_dis=True,
        emb_bond=False,
        cutoff=Length(1,'nm'),
        cutoff_fn='smooth',
        rbf_fn='log_gaussian',
        num_basis=128,
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

    readout = GFNReadout(dim_feature, 128, 'silu', 'relu', 3, Length(1,'nm'), 'smooth', 'nm', ndim=2, shape=(9,3), shared_parms=True)
    net = Cybertron(embedding=emb, model=mod, readout=readout, num_atoms=9, scale=scale*scale2, shift=0)

    outdir = 'Test'
    net.save_configure('configure.yaml', outdir)
    net.print_info()

    tot_params = 0
    for i, param in enumerate(net.get_parameters()):
        tot_params += param.size
        print(i, param.name, param.shape)
    print('Total parameters: ', tot_params)
    
    lr = TransformerLR(0.5,256,dim_feature)
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)

    loss_network = MolWithLossCell(data_keys, net, MSELoss())
    loss_network.print_info()

    eval_network = MolWithEvalCell(data_keys, net, MSELoss(), normed_evaldata=True,add_cast_fp32=True)
    eval_network.print_info()

    eval_mae = 'EvalMAE'
    eval_loss = 'Evalloss'
    eval_rmse = 'EvalRMSE'
    model = Model(loss_network, optimizer=optim, eval_network=eval_network, metrics={eval_mae: MAE(), eval_loss: Loss(), eval_rmse: RMSE()})

    ckpt_name = 'cybertron-' + net.model_name.lower()

    # Using TrainMonitor
    record_cb = TrainMonitor(model, ckpt_name, per_epoch=1, avg_steps=64, directory=outdir, eval_dataset=ds_valid)

    config_ck = CheckpointConfig(save_checkpoint_steps=64, keep_checkpoint_max=8)
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