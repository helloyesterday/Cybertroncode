from cybertroncode.readouts import AtomwiseReadout
import numpy as np
import time
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import dataset as ds
from mindspore.train import Model
from mindspore import context

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint,load_param_into_net
from mindspore.profiler import Profiler

from cybertroncode.units import units
from cybertroncode.models import SchNet,MolCT,PhysNet
from cybertroncode.cybertron import Cybertron
from cybertroncode.train import WithForceLossCell,WithForceEvalCell
from cybertroncode.train import EvalMonitor,MAE,MSE


if __name__ == '__main__':

    # np.set_printoptions(threshold=np.inf)
    seed = 3333
    ms.set_seed(seed)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    bonds = Tensor([[
        [1,1,1,1,-1,-1,-1,-1],
        [1,-1,-1,-1,1,1,1,-1],
        [1,-1,-1,-1,-1,-1,-1,1],
        [1,-1,-1,-1,-1,-1,-1,-1],
        [1,-1,-1,-1,-1,-1,-1,-1],
        [-1,1,-1,-1,-1,-1,-1,-1],
        [-1,1,-1,-1,-1,-1,-1,-1],
        [-1,1,-1,-1,-1,-1,-1,-1],
        [-1,-1,1,-1,-1,-1,-1,-1]],],ms.int32)

    mol_name='ethanol'

    train_file = './' + mol_name + '_train_1024.npz'
    valid_file = './' + mol_name + '_valid_128.npz'
    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    atomic_numbers = Tensor(train_data['z'],ms.int32)
    num_atom = atomic_numbers.size
    graph_scale = float(train_data['scale'][0] / num_atom)
    graph_shift = float(train_data['shift'][0])
    atom_scale = graph_scale
    atom_shift = graph_shift / num_atom

    mod = MolCT(
        min_rbf_dis=0.05,
        max_rbf_dis=10,
        num_rbf=32,
        rbf_sigma=0.2,
        n_interactions=3,
        # interactions=['bond','bond','bond','dis','dis','dis'],
        dim_feature=128,
        n_heads=8,
        max_cycles=1,
        use_time_embedding=True,
        fixed_cycles=True,
        self_dis=0.05,
        unit_length='A',
        use_bonds=False,
        use_feed_forward=False,
        )

    # mod = SchNet(max_rbf_dis=1,num_rbf=32,n_interactions=3,dim_feature=128,dim_filter=128,unit_length='nm')
    # mod = PhysNet(max_rbf_dis=10,num_rbf=32,dim_feature=32,n_interactions=5)

    readout = AtomwiseReadout(n_in=mod.dim_feature,n_out=1,graph_scale=graph_scale,graph_shift=graph_shift,unit_energy='kcal/mol')
    net = Cybertron(mod,node_types=atomic_numbers,full_connect=True,readout=readout,unit_dis='A',unit_energy='kcal/mol')

    network_name = mod.network_name

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
        # print(i,param.asnumpy())
    print('Total parameters: ',tot_params)
    net.print_info()

    n_epoch = 128
    repeat_time = 1
    batch_size = 32
    R = Tensor(valid_data['R'][0:8],units.float)
    E = Tensor(valid_data['E'][0:8],units.float)
    out = net(R)
    for o,e in zip(out,E):
        print(o,e)

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'F':train_data['F'],'E':train_data['E']},shuffle=True)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'F':valid_data['F'],'E':valid_data['E']},shuffle=False)
    ds_valid = ds_valid.batch(len(valid_data['E']))
    ds_valid = ds_valid.repeat(1)

    loss_opeartion = WithForceLossCell('RFE',net,nn.MSELoss(),nn.MSELoss())
    eval_opeartion = WithForceEvalCell('RFE',net)

    decay_rate = np.exp(np.log(0.1)/64)
    lr = nn.ExponentialDecayLR(learning_rate=1e-3, decay_rate=decay_rate, decay_steps=32, is_stair=True)
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr)

    energy_mae = 'EnergyMAE'
    forces_mae = 'ForcesMAE'
    forces_mse = 'ForcesMSE'

    model = Model(loss_opeartion,eval_network=eval_opeartion,optimizer=optim,metrics={energy_mae:MAE([2,3]),forces_mae:MAE([4,5]),forces_mse:MSE([4,5])},amp_level='O0')

    outdir = mol_name + '_' + network_name + '_debug'

    params_name = mol_name + '_' + network_name
    config_ck = CheckpointConfig(save_checkpoint_steps=1024, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=params_name, directory=outdir, config=config_ck)

    record_file = mol_name + '_' + network_name
    record_cb = EvalMonitor(model, record_file, directory=outdir, per_epoch=1, eval_dataset=ds_valid, best_ckpt_metrics=forces_mse)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train,callbacks=[record_cb,ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))
