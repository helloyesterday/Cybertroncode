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
from cybertroncode.train import SquareLoss,AbsLoss,ForceAbsLoss,MLoss
from cybertroncode.train import WithLabelLossCell_RZE,WithLabelEvalCell_RZE
from cybertroncode.train import WithLabelLossCell_RZBE,WithLabelEvalCell_RZBE
from cybertroncode.train import Recorder,MAE,MSE,MAE_per_atom,TransformerLR
from cybertroncode.readouts import AtomwiseReadout,GraphReadout

if __name__ == '__main__':

    # seed = 1111
    # ms.set_seed(seed)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # profiler = Profiler()

    mol_name='qm9_ev_energy4'
    train_file = './' + mol_name + '_train_130724.npz'
    valid_file = './' + mol_name + '_valid_99.npz'
    train_data = np.load(train_file)
    valid_data = np.load(valid_file)

    num_atom = int(train_data['Nmax'])
    scale = Tensor(train_data['atom_std'],ms.float32)
    shift = Tensor(train_data['atom_avg'],ms.float32)
    atom_ref = Tensor(train_data['ref'],ms.float32)

    mod = MolCT(
        min_rbf_dis=0.01,
        max_rbf_dis=2,
        num_rbf=64,
        rbf_sigma=0.2,
        # n_interactions=3,
        interactions=['dis',]*3,
        dim_feature=128,
        n_heads=8,
        max_cycles=1,
        use_time_embedding=True,
        fixed_cycles=True,
        self_dis=0.01,
        unit_length='nm',
        use_feed_forward=False,
        coupled_interactions=False,
        )
    # mod = SchNet(max_rbf_dis=2,num_rbf=64,n_interactions=3,dim_feature=128,dim_filter=128,unit_length='nm',use_graph_norm=True)
    # mod = PhysNet(max_rbf_dis=10,num_rbf=32,dim_feature=32,n_interactions=5)
    readout = AtomwiseReadout(n_in=mod.dim_feature,n_interactions=mod.n_interactions,n_out=[1,1,1,1],activation=[Swish(),Swish(),Swish(),Swish()])
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

    n_epoch = 512
    repeat_time = 1
    batch_size = 32

    # R = Tensor(valid_data['R'][0:16],ms.float32)
    # z = Tensor(valid_data['z'][0:16],ms.int32)
    # Y = Tensor(valid_data['properties'][0:16],ms.float32)

    # out = net(R,z)
    # for o,e in zip(out,Y):
    #     print(o,e)

    ds_train = ds.NumpySlicesDataset({'R':train_data['R'],'z':train_data['z'],'energies':train_data['properties']},shuffle=True)
    ds_train = ds_train.batch(batch_size,drop_remainder=True)
    ds_train = ds_train.repeat(repeat_time)

    ds_valid = ds.NumpySlicesDataset({'R':valid_data['R'],'z':valid_data['z'],'energies':valid_data['properties']},shuffle=False)
    ds_valid = ds_valid.batch(len(valid_data['properties'][0]))
    ds_valid = ds_valid.repeat(1)

    loss_opeartion = WithLabelLossCell_RZE(net,nn.MAELoss(),do_whitening=True,scale=scale,shift=shift,references=atom_ref)
    eval_opeartion = WithLabelEvalCell_RZE(net,nn.MAELoss(),do_scaleshift=True,scale=scale,shift=shift,references=atom_ref)

    # lr = nn.ExponentialDecayLR(learning_rate=1e-3, decay_rate=0.96, decay_steps=8096, is_stair=True)
    lr = TransformerLR(learning_rate=1.,warmup_steps=4000,dimension=128)
    optim = nn.Adam(params=net.trainable_params(),learning_rate=lr,beta1=0.9,beta2=0.98,eps=1e-9)

    valid_mae = 'ValidMAE'
    valid_loss = "ValidLoss"
    # model = Model(train_net,eval_network=eval_opeartion,metrics={output_mae:MAE([1,2],False),avg_mae:MAE([1,2])},amp_level='O3')
    model = Model(loss_opeartion,optimizer=optim,eval_network=eval_opeartion,metrics={valid_mae:MAE([1,2],False),valid_loss:MLoss(0)},amp_level='O0')

    outdir = mol_name + '_' + network_name + '_dis3_white_512'

    params_name = mol_name + '_' + network_name
    config_ck = CheckpointConfig(save_checkpoint_steps=400, keep_checkpoint_max=64)
    ckpoint_cb = ModelCheckpoint(prefix=params_name, directory=outdir, config=config_ck)

    record_file = mol_name + '_' + network_name
    record_cb = Recorder(model, record_file, per_step=200, avg_steps=200, directory=outdir, eval_dataset=ds_valid, dynamic_lr=lr,best_ckpt_metrics=valid_mae)

    print("Start training ...")
    beg_time = time.time()
    model.train(n_epoch,ds_train,callbacks=[record_cb,ckpoint_cb],dataset_sink_mode=False)
    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Training Fininshed!")
    print ("Training Time: %02d:%02d:%02d" % (h, m, s))

    # profiler.analyse()