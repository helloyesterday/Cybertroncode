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
from cybertroncode.readouts import GraphReadout,AtomwiseReadout
from cybertroncode.cybertron import Cybertron
from cybertroncode.train import TransformerLR
from cybertroncode.train import TrainMonitor

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    sys_name = 'ethanol_kcal_A'

    train_file = sys_name + '_train_1024.npz'
    valid_file = sys_name + '_valid_128.npz'
    test_file  = sys_name + '_test_1024.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)
    test_data = np.load(test_file)

    atom_types = Tensor(train_data['z'],ms.int32)
    num_atom = atom_types.size
    mol_scale = float(train_data['std'] / num_atom)
    mol_shift = float(train_data['avg'])

    mod = MolCT(
        min_rbf_dis=0.1,
        max_rbf_dis=10,
        num_rbf=32,
        rbf_sigma=0.2,
        n_interactions=3,
        dim_feature=128,
        n_heads=8,
        max_cycles=1,
        self_dis=0.1,
        unit_length='A',
        )

    readout = AtomwiseReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=1,mol_scale=mol_scale,mol_shift=mol_shift,unit_energy='kcal/mol')
    net = Cybertron(mod,atom_types=atom_types,full_connect=True,readout=readout,unit_dis='A',unit_energy='kcal/mol')

    net.print_info()

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
    print('Total parameters: ',tot_params)

    n_epoch = 32
    repeat_time = 1
    batch_size = 32

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'F':train_data['F'],'E':train_data['E']},shuffle=True)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'F':valid_data['F'],'E':valid_data['E']},shuffle=False)
    ds_valid = ds_valid.batch(128)
    ds_valid = ds_valid.repeat(1)

    ds_test = ds.NumpySlicesDataset({'R':test_data['R'],'F':test_data['F'],'E':test_data['E']},shuffle=False)
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)
    
    from cybertroncode.train import WithForceLossCell,WithForceEvalCell,MSELoss
    loss_network = WithForceLossCell('RFE',net,MSELoss(ratio_energy=1,ratio_forces=100))
    eval_network = WithForceEvalCell('RFE',net)

    lr = nn.ExponentialDecayLR(learning_rate=1e-3, decay_rate=0.96, decay_steps=64, is_stair=True)
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    outdir = 'tutorial_07'
    outname = outdir + mod.network_name

    from cybertroncode.train import MAE,RMSE
    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_rmse = 'ForcesRMSE'
    model = Model(loss_network,eval_network=eval_network,optimizer=optim,metrics={energy_mae:MAE([1,2]),forces_mae:MAE([3,4]),forces_rmse:RMSE([3,4],atom_aggregate='sum')})

    record_cb = TrainMonitor(model, outname, per_epoch=1, avg_steps=32, directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=forces_rmse)

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