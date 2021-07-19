import sys
import numpy as np
import time
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

sys.path.append('..')
from cybertroncode.models import SchNet,MolCT,PhysNet
from cybertroncode.cybertron import Cybertron

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    sys_name = 'qm9_ev_A'

    train_file = sys_name + '_train_1024.npz'
    valid_file = sys_name + '_valid_128.npz'
    test_file  = sys_name + '_test_1024.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)
    test_data = np.load(test_file)
    
    idx = [7] # U0

    num_atom = int(train_data['Nmax'])
    # scale = Tensor(train_data['mol_std'][idx],ms.float32)
    # shift = Tensor(train_data['mol_avg'][idx],ms.float32)
    atom_scale = Tensor(train_data['atom_std'][idx],ms.float32)
    atom_shift = Tensor(train_data['atom_avg'][idx],ms.float32)
    atom_ref = Tensor(train_data['ref'][:,idx],ms.float32)

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

    from cybertroncode.readouts import AtomwiseReadout,GraphReadout
    # readout = GraphReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=1,activation='swish',graph_scale=scale,graph_shift=shift,unit_energy=None)
    readout = AtomwiseReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=1,activation='swish',atom_scale=atom_scale,atom_shift=atom_shift,atom_ref=atom_ref,unit_energy='kcal/mol')
    net = Cybertron(mod,max_atoms_number=num_atom,full_connect=True,readout=readout,unit_dis='A',unit_energy='kcal/mol')

    net.print_info()

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
    print('Total parameters: ',tot_params)

    n_epoch = 8
    repeat_time = 1
    batch_size = 32

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'z':train_data['z'],'E':train_data['properties'][:,idx]},shuffle=True)
    ds_train = ds_train.batch(batch_size,drop_remainder=True)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'z':valid_data['z'],'E':valid_data['properties'][:,idx]},shuffle=False)
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    ds_test = ds.NumpySlicesDataset({'R':test_data['R'],'z':test_data['z'],'E':test_data['properties'][:,idx]},shuffle=False)
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)

    from cybertroncode.train import WithLabelLossCell,WithLabelEvalCell
    loss_network = WithLabelLossCell('RZE',net,nn.MAELoss())
    eval_network = WithLabelEvalCell('RZE',net,nn.MAELoss())

    lr = 1e-3
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    from cybertroncode.train import MAE,MLoss
    eval_mae  = 'EvalMAE'
    atom_mae  = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(loss_network,optimizer=optim,eval_network=eval_network,metrics={eval_mae:MAE([1,2]),atom_mae:MAE([1,2,3],averaged_by_atoms=True),eval_loss:MLoss(0)})

    outdir = 'tutorial_02'
    outname = sys_name + mod.network_name

    from cybertroncode.train import TrainMonitor
    record_cb = TrainMonitor(model, outname, per_step=16, avg_steps=16, directory=outdir, eval_dataset=ds_valid)

    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=outname, directory=outdir, config=config_ck)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train,callbacks=[record_cb,ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))

    print('Test dataset:')
    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k,v in eval_metrics.items():
        info += k
        info += ': '
        info += str(v)
        info += ', '
    print(info)