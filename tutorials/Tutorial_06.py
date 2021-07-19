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
from cybertroncode.train import WithLabelLossCell,WithLabelEvalCell
from cybertroncode.train import TransformerLR
from cybertroncode.train import MAE,MLoss
from cybertroncode.train import TrainMonitor

if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    sys_name = 'qm9_ev_A'

    train_file = sys_name + '_train_1024.npz'
    valid_file = sys_name + '_valid_128.npz'
    test_file  = sys_name + '_test_1024.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)
    test_data = np.load(test_file)
    
    idx1 = [0,1,2,3,4,5,6,11] # diplole,polarizability,HOMO,LUMO,gap,R2,zpve,capacity
    idx2 = [7,8,9,10] # U0,U,G,H
    idx = idx1 + idx2

    num_atom = int(train_data['Nmax'])
    mol_scale = train_data['mol_std'][idx1]
    mol_shift = train_data['mol_avg'][idx1]
    atom_scale = train_data['atom_std'][idx2]
    atom_shift = train_data['atom_avg'][idx2]

    scale1 = np.ones_like(mol_scale)
    shift1 = np.zeros_like(mol_scale)
    scale2 = np.ones_like(atom_scale)
    shift2 = np.zeros_like(atom_scale)

    mol_scale = Tensor(np.concatenate([mol_scale,scale2]),ms.float32)
    mol_shift = Tensor(np.concatenate([mol_shift,shift2]),ms.float32)
    atom_scale = Tensor(np.concatenate([scale1,atom_scale]),ms.float32)
    atom_shift = Tensor(np.concatenate([shift1,atom_shift]),ms.float32)

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

    readout1 = GraphReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=[1,1,3,1,1,1],activation='swish',unit_energy=None)
    readout2 = AtomwiseReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=[1,1,1,1],activation='swish',unit_energy=None)
    net = Cybertron(mod,max_atoms_number=num_atom,full_connect=True,readout=[readout1,readout2],unit_dis='A',unit_energy=None)

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
    
    loss_network = WithLabelLossCell('RZE',net,nn.MAELoss(),do_whitening=True,mol_scale=mol_scale,mol_shift=mol_shift,atom_scale=atom_scale,atom_shift=atom_shift,atom_ref=atom_ref)
    eval_network = WithLabelEvalCell('RZE',net,nn.MAELoss(),do_whitening=True,mol_scale=mol_scale,mol_shift=mol_shift,atom_scale=atom_scale,atom_shift=atom_shift,atom_ref=atom_ref)

    lr = TransformerLR(learning_rate=1.,warmup_steps=4000,dimension=128)
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    outdir = 'tutorial_06'
    outname = outdir + mod.network_name

    eval_mae  = 'EvalMAE'
    atom_mae  = 'AtomMAE'
    eval_loss = 'Evalloss'
    model = Model(loss_network,optimizer=optim,eval_network=eval_network,metrics={eval_mae:MAE([1,2],reduce_all_dims=False),atom_mae:MAE([1,2,3],reduce_all_dims=False,averaged_by_atoms=True),eval_loss:MLoss(0)})

    record_cb = TrainMonitor(model, outname, per_step=16, avg_steps=16, directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=eval_loss)

    config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=outname, directory=outdir, config=config_ck)

    np.set_printoptions(linewidth=300)

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