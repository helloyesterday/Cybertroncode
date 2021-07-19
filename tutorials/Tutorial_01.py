import sys
import numpy as np
import time
import mindspore as ms
from mindspore import nn
from mindspore import Tensor

sys.path.append('..')

if __name__ == '__main__':

    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    train_file = 'qm9_ev_A_train_1024.npz'

    train_data = np.load(train_file)
    
    idx = [0] # diple

    num_atom = int(train_data['Nmax'])
    scale = Tensor(train_data['mol_std'][idx],ms.float32)
    shift = Tensor(train_data['mol_avg'][idx],ms.float32)

    from cybertroncode.models import SchNet,MolCT,PhysNet

    mod = MolCT(
        min_rbf_dis=0.1,
        max_rbf_dis=20,
        num_rbf=64,
        n_interactions=3,
        dim_feature=128,
        n_heads=8,
        activation='swish',
        max_cycles=1,
        unit_length='A',
        )

    # mod = SchNet(
    #     max_rbf_dis=2,
    #     num_rbf=64,
    #     n_interactions=3,
    #     activation='shifted',
    #     dim_feature=128,
    #     dim_filter=128,
    #     unit_length='A',
    #     )

    # mod = PhysNet(
    #     max_rbf_dis=10,
    #     num_rbf=32,
    #     dim_feature=32,
    #     n_interactions=5,
    #     activation='swish',
    #     )

    from cybertroncode.readouts import GraphReadout
    readout = GraphReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=1,activation='swish',mol_scale=scale,mol_shift=shift,unit_energy=None)

    from cybertroncode.cybertron import Cybertron
    net = Cybertron(mod,max_atoms_number=num_atom,full_connect=True,readout=readout,unit_dis='A')

    net.print_info()

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
    print('Total parameters: ',tot_params)

    n_epoch = 8
    repeat_time = 1
    batch_size = 32

    from mindspore import dataset as ds
    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'z':train_data['z'],'E':train_data['properties'][:,idx]},shuffle=True)
    ds_train = ds_train.batch(batch_size,drop_remainder=True)
    ds_train = ds_train.repeat(repeat_time)

    from cybertroncode.train import WithLabelLossCell
    loss_network = WithLabelLossCell('RZE',net,nn.MAELoss())

    lr = 1e-3
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    from mindspore.train import Model
    model = Model(loss_network,optimizer=optim)

    from mindspore.train.callback import LossMonitor
    monitor_cb = LossMonitor(16)

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
    outdir = 'tutorial_01'
    params_name = outdir + mod.network_name
    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=params_name, directory=outdir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    # model.train(n_epoch,ds_train,dataset_sink_mode=False)
    # model.train(n_epoch,ds_train,callbacks=monitor_cb,dataset_sink_mode=False)
    model.train(n_epoch,ds_train,callbacks=[monitor_cb,ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))