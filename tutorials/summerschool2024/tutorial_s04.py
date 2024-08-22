# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
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
Training discriminator.
"""

import sys
import numpy as np
import time
import h5py
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore.ops import functional as F
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore import context

import sys
import os
path = os.getenv('MINDSPONGE_HOME')
if path:
    sys.path.insert(0, path)
sys.path.append('../..')
data_dir = '/home/yuansh/cybertron/tutorials/summerschool2024/data'

from cybertron.model import MolCT
from cybertron.embedding import MolEmbedding
from cybertron.readout import AtomwiseReadout
from cybertron.cybertron import Cybertron
from cybertron.train import TransformerLR
from cybertron.train import TrainMonitor
from cybertron.train import WithAdversarialLossCell
from cybertron.train import CrossEntropyLoss
from cybertron.train.metric import MAE, RMSE, Loss

class BCELossForDiscriminator(nn.Cell):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.cross_entropy = nn.BCEWithLogitsLoss(reduction)

    def construct(self, pos_pred: Tensor, neg_pred: Tensor):
        """calculate cross entropy loss function

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        pos_loss = self.cross_entropy(pos_pred, F.ones_like(pos_pred))
        neg_loss = self.cross_entropy(neg_pred, F.zeros_like(neg_pred))

        return pos_loss + neg_loss, None, None, None

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Set the backend.", default="GPU")
    args = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.e)

    ori_train_set = data_dir + '/dataset/data_normed_trainset_83197_64_64.npz'
    ori_train_set = np.load(ori_train_set)
    Z = ori_train_set['atom_type']
    pos_data = ori_train_set['coordinate']

    atom_type = Tensor(Z,ms.int32)
    num_atom = int(atom_type.shape[-1])

    data_num = pos_data.shape[0]
    traj_file = data_dir + '/traj/PES_4-100000-800K-bias-NORMAL2.h5md'
    traj = h5py.File(traj_file)['particles/trajectory0/position/value']
    neg_data = np.array(traj,dtype=np.float32)[np.random.choice(traj.shape[0],data_num,replace=False)]

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

    net.print_info()
    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
    print('Total parameters: ',tot_params)

    n_epoch = 40
    repeat_time = 1
    batch_size = 64
    val_size = 32

    idx = np.random.choice(np.arange(data_num), val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(data_num), idx)

    ds_train_dsc = ds.NumpySlicesDataset({'pos':pos_data[train_idx],'neg':neg_data[train_idx]},shuffle=True)
    ds_val_dsc = ds.NumpySlicesDataset({'pos':pos_data[idx],'neg':neg_data[idx]},shuffle=True)
    ds_train_dsc = ds_train_dsc.batch(batch_size)
    ds_train_dsc = ds_train_dsc.repeat(repeat_time)
    ds_val_dsc = ds_val_dsc.batch(batch_size)
    ds_val_dsc = ds_val_dsc.repeat(repeat_time)
    loss_fn = BCELossForDiscriminator()
    loss_network = WithAdversarialLossCell(net,loss_fn)
    eval_network = WithAdversarialLossCell(net,loss_fn)

    lr = TransformerLR(learning_rate=.3, warmup_steps=8000, dimension=128) # smaller
    optim = nn.Adam(params=net.trainable_params(), learning_rate=lr)
    eval_loss = 'EvalLoss'
    neg_loss = 'NegLoss'
    model = Model(loss_network,eval_network=eval_network, optimizer=optim,metrics={eval_loss:Loss()}) 
    model.eval(ds_val_dsc,dataset_sink_mode=False)

    conf_dir = data_dir + '/conf'
    net.save_configure('configure_discr' + '.yaml' , conf_dir)
    outname = 'discr_' + net.model_name
    ckpt_dir = data_dir + '/ckpt'
    record_cb = TrainMonitor(model, outname, per_epoch=1, avg_steps=32, directory=ckpt_dir, eval_dataset=ds_val_dsc, best_ckpt_metrics=eval_loss)
    config_ck = CheckpointConfig(save_checkpoint_steps=8, keep_checkpoint_max=8)
    ckpoint_cb = ModelCheckpoint(prefix=outname, directory=ckpt_dir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train_dsc,callbacks=[record_cb, ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))

