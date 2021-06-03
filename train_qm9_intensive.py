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
from mindspore.train.callback import LossMonitor,SummaryCollector

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig,Callback
from mindspore.train.serialization import load_checkpoint,load_param_into_net
from mindspore.profiler import Profiler

from cybertroncode.models import SchNet,MolCT,PhysNet
from cybertroncode.cybertron import Cybertron
from cybertroncode.cutoff import MollifierCutoff
from cybertroncode.rbf import LogGaussianDistribution
from cybertroncode.activations import ShiftedSoftplus,Swish
from cybertroncode.cutoff import CosineCutoff,SmoothCutoff
from cybertroncode.train import MLoss
from cybertroncode.train import WithLabelLossCell,WithLabelEvalCell
from cybertroncode.train import EvalMonitor,MAE,MSE,MAEAveragedByAtoms,TransformerLR
from cybertroncode.readouts import AtomwiseReadout,GraphReadout

if __name__ == '__main__':

    # seed = 1111
    # ms.set_seed(seed)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    mol_name='qm9_ev_A'
    num = 130823
    n_train = 102400
    n_valid = 2048
    n_test = num - n_train - n_valid
    train_file = './' + mol_name + '_train_' + str(n_train) + '.npz'
    valid_file = './' + mol_name + '_valid_' + str(n_valid) + '.npz'
    test_file = './' + mol_name + '_test_' + str(n_test) + '.npz'

    train_data = np.load(train_file)
    valid_data = np.load(valid_file)
    test_data = np.load(test_file)

    idx = [0,2,3,4,11]

    num_atom = int(train_data['Nmax'])
    scale = Tensor(train_data['mol_std'][idx],ms.float32)
    shift = Tensor(train_data['mol_avg'][idx],ms.float32)

    mod = MolCT(
        min_rbf_dis=0.1,
        max_rbf_dis=20,
        num_rbf=64,
        rbf_sigma=0.2,
        # n_interactions=3,
        interactions=['dis',]*3,
        dim_feature=128,
        n_heads=8,
        max_cycles=1,
        use_time_embedding=True,
        fixed_cycles=True,
        self_dis=0.1,
        unit_length='A',
        use_feed_forward=False,
        coupled_interactions=False,
        )
    # mod = SchNet(max_rbf_dis=2,num_rbf=64,n_interactions=3,dim_feature=128,dim_filter=128,unit_length='nm',use_graph_norm=True)
    # mod = PhysNet(max_rbf_dis=10,num_rbf=32,dim_feature=32,n_interactions=5)
    
    readout = GraphReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=[1,3,1],activation='swish',aggregator='transformer')
    # net = Cybertron(mod,max_nodes_number=num_atom,full_connect=True,atom_scale=scale,atom_shift=shift,atom_ref=atom_ref,readout='atom',unit_dis='A',unit_energy='eV',dim_output=4,)
    net = Cybertron(mod,max_nodes_number=num_atom,full_connect=True,readout=readout,unit_dis='A')

    network_name = mod.network_name
    net.print_info()

    tot_params = 0
    for i,param in enumerate(net.trainable_params()):
        tot_params += param.size
        print(i,param.name,param.shape)
        # print(i,param.asnumpy())
    print('Total parameters: ',tot_params)
    # print(net)

    print(scale,shift)

    n_epoch = 1024
    repeat_time = 1
    batch_size = 32

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'z':train_data['z'],'energies':train_data['properties'][:,idx]},shuffle=True)
    ds_train = ds_train.batch(batch_size,drop_remainder=True)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'z':valid_data['z'],'energies':valid_data['properties'][:,idx]},shuffle=False)
    ds_valid = ds_valid.batch(n_valid)
    ds_valid = ds_valid.repeat(1)

    loss_network = WithLabelLossCell('RZE',net,nn.MAELoss(),do_whitening=True,scale=scale,shift=shift)
    eval_network = WithLabelEvalCell('RZE',net,nn.MAELoss(),do_scaleshift=True,scale=scale,shift=shift)

    # lr = nn.ExponentialDecayLR(learning_rate=1e-3, decay_rate=0.96, decay_steps=8096, is_stair=True)
    lr = TransformerLR(learning_rate=1.,warmup_steps=4000,dimension=128)
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr,beta1=0.9,beta2=0.98,eps=1e-9)

    valid_mae = 'EvalMAE'
    valid_loss = "EvalLoss"
    model = Model(loss_network,optimizer=optim,eval_network=eval_network,metrics={valid_mae:MAE([1,2],False),valid_loss:MLoss(0)},amp_level='O0')

    outdir = mol_name + '_' + network_name + '_n' + str(n_test) + '_dis3_white_1024'

    params_name = mol_name + '_' + network_name
    config_ck = CheckpointConfig(save_checkpoint_steps=3200, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=params_name, directory=outdir, config=config_ck)

    record_file = mol_name + '_' + network_name
    record_cb = EvalMonitor(model, record_file, per_step=50, avg_steps=400, directory=outdir, eval_dataset=ds_valid, best_ckpt_metrics=valid_mae)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train,callbacks=[record_cb,ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))

    ds_test = ds.NumpySlicesDataset({'R':test_data['R'],'z':test_data['z'],'energies':test_data['properties'][:,idx]},shuffle=False)
    ds_test = ds_test.batch(1024)
    ds_test = ds_test.repeat(1)

    eval_metrics = model.eval(ds_test, dataset_sink_mode=False)
    info = ''
    for k,v in eval_metrics.items():
        info += k
        info += ': '
        info += str(v)
        info += ', '
    print('Test dataset: '+info)