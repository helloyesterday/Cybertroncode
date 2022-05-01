# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
# 
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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

import os
import numpy as np
import mindspore as ms
import mindspore.numpy as msnp
from shutil import copyfile
from collections import deque
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.nn import TrainOneStepCell
from mindspore.train.callback import Callback,RunContext
from mindspore.train.callback._callback import InternalCallbackParam
from mindspore.nn.metrics import Metric
from mindspore.train._utils import _make_directory
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore._checkparam import Validator as validator

from sponge import hyperparam

from cybertron.cybertron import Cybertron

from sponge.checkpoint import save_checkpoint

_cur_dir = os.getcwd()

__all__ = [
    "EvalScaleShift",
    "WithForceLossCell",
    "WithLabelLossCell",
    "WithForceEvalCell",
    "WithLabelEvalCell",
    "TrainMonitor",
    "MAE",
    "MSE",
    "MAEAveragedByAtoms",
    "MLoss",
    "TransformerLR",
    ]

class OutputScaleShift(Cell):
    def __init__(
        self,
        scale=1,
        shift=0,
        type_ref=None,
        atomwise_scaleshift=None,
        axis=-2,
    ):
        super().__init__()

        self.scale = Tensor(scale,ms.float32)
        self.shift = Tensor(shift,ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref,ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift,ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True
        
        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(self.atomwise_scaleshift,(1,-1))

        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, outputs:Tensor, num_atoms:Tensor, atom_types:Tensor=None):
        ref = 0
        if self.type_ref is not None:
            ref = F.gather(self.type_ref,atom_types,0)
            ref = self.reduce_sum(ref,self.axis)
        
        outputs = outputs * self.scale + ref
        if self.all_atomwsie:
            return outputs + self.shift * num_atoms
        elif self.all_graph:
            return outputs + self.shift
        else:
            atomwise_output = outputs + self.shift * num_atoms
            graph_output = outputs + self.shift
            return msnp.where(self.atomwise_scaleshift,atomwise_output,graph_output)

class DatasetNormalization(Cell):
    def __init__(
        self,
        scale=1,
        shift=0,
        type_ref=None,
        atomwise_scaleshift=None,
        axis=-2,
    ):
        super().__init__()

        self.scale = Tensor(scale,ms.float32)
        self.shift = Tensor(shift,ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref,ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift,ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True
        
        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(self.atomwise_scaleshift,(1,-1))

        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, label, num_atoms, atom_types=None):
        ref = 0
        if self.type_ref is not None:
            ref = F.gather(self.type_ref,atom_types,0)
            ref = self.reduce_sum(ref,self.axis)

        label -= ref
        if self.all_atomwsie:
            return (label - self.shift * num_atoms) / self.scale
        elif self.all_graph:
            return (label - self.shift) / self.scale
        else:
            atomwise_norm = (label - self.shift * num_atoms) / self.scale
            graph_norm = (label - self.shift) / self.scale
            return msnp.where(self.atomwise_scaleshift,atomwise_norm,graph_norm)

class LossWithEnergyAndForces(nn.loss.loss.LossBase):
    def __init__(self,
        ratio_energy=1,
        ratio_forces=100,
        force_dis=1,
        ratio_normlize=True,
        reduction='mean',
    ):
        super().__init__(reduction)

        self.force_dis = Tensor(force_dis,ms.float32)
        self.ratio_normlize = ratio_normlize

        self.ratio_energy = ratio_energy
        self.ratio_forces = ratio_forces

        self.norm = 1
        if self.ratio_normlize:
            self.norm = ratio_energy + ratio_forces

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def _calc_loss(self,diff):
        return diff

    def construct(self, pred_energy, label_energy, pred_forces=None, label_forces=None,  num_atoms=1, atom_mask=None):

        if pred_forces is None:
            loss = self._calc_loss(pred_energy - label_energy)
            return self.get_loss(loss)

        eloss = 0
        if self.ratio_forces > 0:
            ediff = (pred_energy - label_energy) / num_atoms
            eloss = self._calc_loss(ediff)

        floss = 0
        if self.ratio_forces > 0:
            # (B,A,D)
            fdiff = (pred_forces - label_forces) * self.force_dis
            fdiff = self._calc_loss(fdiff)
            # (B,A)
            fdiff = self.reduce_sum(fdiff,-1)

            if atom_mask is None:
                floss = self.reduce_mean(fdiff,-1)
            else:
                fdiff = fdiff * atom_mask
                floss = self.reduce_sum(fdiff,-1)
                floss = floss / num_atoms

        y = (eloss * self.ratio_energy + floss * self.ratio_forces) / self.norm

        natoms = F.cast(num_atoms,pred_energy.dtype)
        weights = natoms / self.reduce_mean(natoms)

        return self.get_loss(y,weights)

class MAELoss(LossWithEnergyAndForces):
    def __init__(self,
        ratio_energy=1,
        ratio_forces=0,
        force_dis=1,
        ratio_normlize=True,
        reduction='mean',
    ):
        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )
        self.abs = P.Abs()
    
    def _calc_loss(self, diff):
        return self.abs(diff)

class MSELoss(LossWithEnergyAndForces):
    def __init__(self,
        ratio_energy=1,
        ratio_forces=0,
        force_dis=1,
        ratio_normlize=True,
        reduction='mean',
    ):
        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )
        self.square = P.Square()
    
    def _calc_loss(self, diff):
        return self.square(diff)

class CrossEntropyLoss(nn.loss.loss.LossBase):
    def __init__(self, reduction='mean', use_sigmoid=False):
        super().__init__(reduction)
        
        self.sigmoid = None
        if use_sigmoid:
            self.sigmoid = P.Sigmoid()

        self.cross_entropy = P.BinaryCrossEntropy(reduction)

    def construct(self, pos_pred, neg_pred):
        if self.sigmoid is not None:
            pos_pred = self.sigmoid(pos_pred)
            neg_pred = self.sigmoid(neg_pred)

        pos_loss = self.cross_entropy(pos_pred,F.ones_like(pos_pred))
        neg_loss = self.cross_entropy(neg_pred,F.zeros_like(neg_pred))

        return  pos_loss + neg_loss

class WithCell(Cell):
    def __init__(self,
        datatypes: str,
        network: Cybertron,
        loss_fn: Cell,
        fulltypes: str='RZCNnBbE',
        cell_name: str='',
    ):
        super().__init__(auto_prefix=False)

        self.fulltypes = fulltypes
        self.datatypes = datatypes

        if not isinstance(self.datatypes,str):
            raise TypeError('Type of "datatypes" must be str')

        for datatype in self.datatypes:
            if self.fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in self.fulltypes:
            num = self.datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype + '" in datatype "' + self.datatypes + '".')

        self.R = self.datatypes.find('R') # positions
        self.Z = self.datatypes.find('Z') # atom_types
        self.C = self.datatypes.find('C') # pbcbox
        self.N = self.datatypes.find('N') # neighbours
        self.n = self.datatypes.find('n') # neighbour_mask
        self.B = self.datatypes.find('B') # bonds
        self.b = self.datatypes.find('b') # bond_mask
        self.E = self.datatypes.find('E') # energy

        if self.E < 0:
            raise TypeError('The datatype "E" must be included!')

        self._network = network
        self._loss_fn = loss_fn

        self.hyper_param = None
        if 'hyper_param' in self._network.__dict__.keys():
            self.hyper_param = self._network.hyper_param

        if 'atom_types' in self._network.__dict__.keys():
            self.atom_types = self._network.atom_types

        print(self.cls_name + ' with input type: ' + self.datatypes)

        self.keep_sum = P.ReduceSum(keep_dims=True)

class WithForceLossCell(WithCell):
    def __init__(self,
        datatypes: str,
        network: Cybertron,
        loss_fn: Cell,
    ):
        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCNnBbFE'
        )

        self.F = self.datatypes.find('F') # force
        if self.F < 0:
            raise TypeError('The datatype "F" must be included in WithForceLossCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbcbox = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        energy = inputs[self.E]
        out = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        forces = inputs[self.F]
        fout = -1 * self.grad_op(self._network)(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types>0,out.dtype)
        num_atoms = self.keep_sum(num_atoms,-1)

        if atom_types is None:
            return self._loss_fn(out,energy,fout,forces)
        else:
            atom_mask = atom_types > 0
            return self._loss_fn(out,energy,fout,forces,num_atoms,atom_mask)

    @property
    def backbone_network(self):
        return self._network

class WithLabelLossCell(WithCell):
    def __init__(self,
        datatypes: str,
        network: Cybertron,
        loss_fn: Cell,
        # with_penalty=False,
    ):
        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCNnBbE'
        )
        # self.with_penalty = with_penalty

    def construct(self, *inputs):

        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbcbox = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        out = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        label = inputs[self.E]

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types>0,out.dtype)
        num_atoms = self.keep_sum(num_atoms,-1)

        return self._loss_fn(out,label)


class WithEvalCell(WithCell):
    def __init__(self,
        datatypes: str,
        network: Cybertron,
        loss_fn: Cell=None,
        scale: float=None,
        shift: float=None,
        type_ref: Tensor=None,
        atomwise_scaleshift: Tensor=None,
        eval_data_is_normed=True,
        add_cast_fp32=False,
        fulltypes='RZCNnBbE'
    ):
        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes=fulltypes
        )
        
        self.scale = scale
        self.shift = shift

        if atomwise_scaleshift is None:
            atomwise_scaleshift = self._network.atomwise_scaleshift
        else:
            atomwise_scaleshift = Tensor(atomwise_scaleshift,ms.bool_)
        self.atomwise_scaleshift = atomwise_scaleshift

        self.scaleshift = None
        self.normalization = None
        self.scaleshift_eval = eval_data_is_normed
        self.normalize_eval = False
        self.type_ref = None
        if scale is not None or shift is not None:
            if scale is None:
                scale = 1
            if shift is None:
                shift = 0

            if type_ref is not None:
                self.type_ref = Tensor(type_ref,ms.float32)

            self.scaleshift = OutputScaleShift(
                scale=scale,
                shift=shift,
                type_ref=self.type_ref,
                atomwise_scaleshift=atomwise_scaleshift
            )

            if self._loss_fn is not None:
                self.normalization = DatasetNormalization(
                    scale=scale,
                    shift=shift,
                    type_ref=self.type_ref,
                    atomwise_scaleshift=atomwise_scaleshift
                )
                if not eval_data_is_normed:
                    self.normalize_eval = True

            self.scale = self.scaleshift.scale
            self.shift = self.scaleshift.shift

            scale = self.scale.asnumpy().reshape(-1)
            shift = self.shift.asnumpy().reshape(-1)
            atomwise_scaleshift = self.scaleshift.atomwise_scaleshift.asnumpy().reshape(-1)
            print('   with scaleshift for training '+
                ('and evaluate ' if eval_data_is_normed else ' ')+'dataset:')
            if atomwise_scaleshift.size == 1:
                print('   Scale: '+str(scale))
                print('   Shift: '+str(shift))
                print('   Scaleshift mode: '+('atomwise' if atomwise_scaleshift else 'graph'))
            else:
                print('   {:>6s}. {:>16s}{:>16s}{:>12s}'.format('Output','Scale','Shift','Mode'))
                for i,m in enumerate(atomwise_scaleshift):
                    _scale = scale if scale.size == 1 else scale[i]
                    _shift = scale if shift.size == 1 else shift[i]
                    mode = 'Atomwise' if m else 'graph'
                    print('   {:<6s}{:>16.6e}{:>16.6e}{:>12s}'.format(str(i)+': ',_scale,_shift,mode))
            if type_ref is not None:
                print('   with reference value for atom types:')
                info = '   Type '
                for i in range(self.type_ref.shape[-1]):
                    info += '{:>10s}'.format('Label'+str(i))
                print(info)
                for i,ref in enumerate(self.type_ref):
                    info = '   {:<7s} '.format(str(i)+':')
                    for r in ref:
                        info += '{:>10.2e}'.format(r.asnumpy())
                    print(info)

        self.add_cast_fp32 = add_cast_fp32
        self.reducesum = P.ReduceSum(keep_dims=True)

class WithLabelEvalCell(WithEvalCell):
    def __init__(self,
        datatypes: str,
        network: Cybertron,
        loss_fn: Cell=None,
        scale: float=None,
        shift: float=None,
        type_ref: Tensor=None,
        atomwise_scaleshift: Tensor=None,
        eval_data_is_normed=True,
        add_cast_fp32=False,
    ):
        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCNnBbE',
        )

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbcbox = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        label = inputs[self.E]
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(ms.float32, label)
            output = F.cast(output, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types>0,ms.int32)
        num_atoms = msnp.sum(atom_types>0,-1,keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            if self.normalize_eval:
                normed_label = self.normalization(label, num_atoms, atom_types)
                loss = self._loss_fn(output, normed_label)
            else:
                loss = self._loss_fn(output, label)

        if self.scaleshift is not None:
            output = self.scaleshift(output, num_atoms, atom_types)
            if self.scaleshift_eval:
                label = self.scaleshift(label, num_atoms, atom_types)

        return loss, output, label, num_atoms


class WithForceEvalCell(WithEvalCell):
    def __init__(self,
        datatypes,
        network: Cybertron,
        loss_fn: Cell=None,
        scale: float=None,
        shift: float=None,
        type_ref: Tensor=None,
        atomwise_scaleshift: Tensor=None,
        eval_data_is_normed: bool=True,
        add_cast_fp32: bool=False,
    ):
        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCNnBbFE',
        )

        self.F = self.datatypes.find('F') # force

        if self.F < 0:
            raise TypeError('The datatype "F" must be included in WithForceEvalCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbcbox = inputs[self.C]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output_energy = self._network(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        output_forces = -1 * self.grad_op(self._network)(
            positions,
            atom_types,
            pbcbox,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        label_forces = inputs[self.F]
        label_energy = inputs[self.E]

        if self.add_cast_fp32:
            label_forces = F.mixed_precision_cast(ms.float32, label_forces)
            label_energy = F.mixed_precision_cast(ms.float32, label_energy)
            output_energy = F.cast(output_energy, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types>0,ms.int32)
        num_atoms = msnp.sum(atom_types>0,-1,keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            atom_mask = atom_types > 0
            if self.normalize_eval:
                normed_label_energy = self.normalization(label_energy, num_atoms, atom_types)
                normed_label_forces = label_forces / self.scale
                loss = self._loss_fn(output_energy, normed_label_energy, output_forces, normed_label_forces, num_atoms, atom_mask)
            else:
                loss = self._loss_fn(output_energy, label_energy, output_forces, label_forces, num_atoms, atom_mask)

        if self.scaleshift is not None:
            output_energy = self.scaleshift(output_energy,num_atoms,atom_types)
            output_forces = output_forces * self.scale
            if self.scaleshift_eval:
                label_energy = self.scaleshift(label_energy, num_atoms, atom_types)
                label_forces = label_forces * self.scale
        
        return loss, output_energy, label_energy, output_forces, label_forces, num_atoms

class WithAdversarialLossCell(Cell):
    def __init__(self,
        network: Cell,
        loss_fn: Cell,
    ):
        super().__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, pos_samples, neg_samples):
        pos_pred = self._network(pos_samples)
        neg_pred = self._network(neg_samples)
        return self._loss_fn(pos_pred, neg_pred)

    @property
    def backbone_network(self):
        return self._network

class TrainMonitor(Callback):
    def __init__(self, model, name, directory=None, per_epoch=1, per_step=0, avg_steps=0, eval_dataset=None, best_ckpt_metrics=None):
        super().__init__()
        if not isinstance(per_epoch, int) or per_epoch < 0:
            raise ValueError("per_epoch must be int and >= 0.")
        if not isinstance(per_step, int) or per_step < 0:
            raise ValueError("per_step must be int and >= 0.")

        self.avg_steps = avg_steps
        self.loss_record = 0
        self.train_num = 0
        if avg_steps > 0:
            self.train_num = deque(maxlen=avg_steps)
            self.loss_record = deque(maxlen=avg_steps)

        if per_epoch * per_step !=0:
            if per_epoch == 1:
                per_epoch = 0
            else:
                raise ValueError("per_epoch and per_step cannot larger than 0 at same time.")
        self.model = model
        self._per_epoch = per_epoch
        self._per_step = per_step
        self.eval_dataset = eval_dataset

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        self._filename = name + '-info.data'
        self._ckptfile = name + '-best'
        self._ckptdata = name + '-ckpt.data'

        self.num_ckpt = 1
        self.best_value = 5e4
        self.best_ckpt_metrics = best_ckpt_metrics

        self.last_loss = 0
        self.record = []

        self.hyper_param = None

        self.output_title = True
        filename = os.path.join(self._directory, self._filename)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(filename)

    def begin(self, run_context):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params:InternalCallbackParam = run_context.original_args()
        train_network:TrainOneStepCell = cb_params.train_network
        cells = train_network._cells
        if 'network' in cells.keys() and 'hyper_param' in cells['network'].__dict__.keys():
            self.hyper_param = cells['network'].hyper_param

    def _write_ckpt_file(self,filename:str,info:str,network:TrainOneStepCell):
        ckptfile = os.path.join(self._directory, filename + '.ckpt')
        ckptbck = os.path.join(self._directory, filename + '.bck.ckpt')
        ckptdata = os.path.join(self._directory, self._ckptdata)

        if os.path.exists(ckptfile):
            os.rename(ckptfile,ckptbck)
        
        save_checkpoint(network,ckptfile,append_dict=self.hyper_param)
        with open(ckptdata, "a") as f:
            f.write(info + os.linesep)

    def _output_data(self,cb_params):
        cur_epoch = cb_params.cur_epoch_num

        opt = cb_params.optimizer
        if opt is None:
            opt = cb_params.train_network.optimizer
        
        if opt.dynamic_lr:
            step = opt.global_step
            if not isinstance(step,int):
                step = step.asnumpy()[0]
        else:
            step = cb_params.cur_step_num

        if self.avg_steps > 0:
            mov_avg = sum(self.loss_record) / sum(self.train_num)
        else:
            mov_avg = self.loss_record / self.train_num
        
        title = "#! FIELDS step"
        info = 'Epoch: ' + str(cur_epoch) + ', Step: ' + str(step)
        outdata = '{:>10d}'.format(step)

        lr = opt.learning_rate
        if opt.dynamic_lr:
            step = F.cast(step,ms.int32)
            if opt.is_group_lr:
                lr = ()
                for learning_rate in opt.learning_rate:
                    current_dynamic_lr = learning_rate(step-1)
                    lr += (current_dynamic_lr,)
            else:
                lr = opt.learning_rate(step-1)
        lr = lr.asnumpy()

        title += ' learning_rate'
        info += ', Learning_rate: ' + str(lr)
        outdata += '{:>15e}'.format(lr)

        title += " last_loss avg_loss"
        info += ', Last_Loss: ' + str(self.last_loss) + ', Avg_loss: ' + str(mov_avg)
        outdata += '{:>15e}'.format(self.last_loss) + '{:>15e}'.format(mov_avg)

        _make_directory(self._directory)

        if self.eval_dataset is not None:
            eval_metrics = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            for k,v in eval_metrics.items():
                info += ', '
                info += k
                info += ': '
                info += str(v)

                if isinstance(v,np.ndarray) and v.size > 1:
                    for i in range(v.size):
                        title += (' ' + k + str(i))
                        outdata += '{:>15e}'.format(v[i])
                else:
                    title += (' ' + k)
                    outdata += '{:>15e}'.format(v)

            if self.best_ckpt_metrics in eval_metrics.keys():
                vnow = eval_metrics[self.best_ckpt_metrics]
                if type(vnow) is np.ndarray and len(vnow) > 1:
                    output_ckpt = vnow < self.best_value
                    num_best = np.count_nonzero(output_ckpt)
                    if num_best > 0:
                        self._write_ckpt_file(self._ckptfile,info,cb_params.train_network)
                        source_ckpt = os.path.join(self._directory, self._ckptfile + '.ckpt')
                        for i in range(len(vnow)):
                            if output_ckpt[i]:
                                dest_ckpt = os.path.join(self._directory, self._ckptfile + '-' + str(i) + '.ckpt')
                                bck_ckpt = os.path.join(self._directory, self._ckptfile + '-' + str(i) + '.ckpt.bck')
                                if os.path.exists(dest_ckpt):
                                    os.rename(dest_ckpt,bck_ckpt)
                                copyfile(source_ckpt,dest_ckpt)
                        self.best_value = np.minimum(vnow,self.best_value)
                else:
                    if vnow < self.best_value:
                        self._write_ckpt_file(self._ckptfile,info,cb_params.train_network)
                        self.best_value = vnow

        print(info, flush=True)
        filename = os.path.join(self._directory, self._filename)
        if self.output_title:
            with open(filename, "a") as f:
                f.write(title + os.linesep)
            self.output_title = False
        with open(filename, "a") as f:
            f.write(outdata + os.linesep)

    def step_end(self, run_context:RunContext):
        cb_params:InternalCallbackParam = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        nbatch = len(cb_params.train_dataset_element[0])
        batch_loss = loss * nbatch

        self.last_loss = loss
        if self.avg_steps > 0:
            self.loss_record.append(batch_loss)
            self.train_num.append(nbatch)
        else:
            self.loss_record += batch_loss
            self.train_num += nbatch

        if self._per_step > 0 and cb_params.cur_step_num % self._per_step == 0:
            self._output_data(cb_params)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if self._per_epoch > 0 and cur_epoch % self._per_epoch == 0:
            self._output_data(cb_params)

class MaxError(Metric):
    def __init__(self,indexes=[1,2],reduce_all_dims=True):
        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._max_error = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        diff = y.reshape(y_pred.shape) - y_pred
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error

class Error(Metric):
    def __init__(self,
        indexes=[1,2],
        reduce_all_dims=True,
        averaged_by_atoms=False,
        atom_aggregate='mean',
    ):
        super().__init__()
        self.clear()
        self._indexes = indexes
        self.read_num_atoms = False
        if len(self._indexes) > 2:
            self.read_num_atoms = True

        self.reduce_all_dims = reduce_all_dims

        if atom_aggregate.lower() not in ('mean','sum'):
            raise ValueError('aggregate_by_atoms method must be "mean" or "sum"')
        self.atom_aggregate = atom_aggregate.lower()

        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

        if averaged_by_atoms and not self.read_num_atoms:
            raise ValueError('When to use averaged_by_atoms, the index of atom number must be set at "indexes".')

        self.averaged_by_atoms = averaged_by_atoms

        self._error_sum = 0
        self._samples_num = 0

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    def _calc_error(self,y,y_pred):
        return y.reshape(y_pred.shape) - y_pred

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])

        error = self._calc_error(y,y_pred)
        if len(error.shape) > 2:
            axis = tuple(range(2,len(error.shape)))
            if self.atom_aggregate == 'mean':
                error = np.mean(error,axis=axis)
            else:
                error = np.sum(error,axis=axis)

        tot = y.shape[0]
        if self.read_num_atoms:
            natoms = self._convert_data(inputs[self._indexes[2]])
            if self.averaged_by_atoms:
                error /= natoms
            elif self.reduce_all_dims:
                tot = np.sum(natoms)
                if natoms.shape[0] != y.shape[0]:
                    tot *= y.shape[0]
        elif self.reduce_all_dims:
            tot = error.size

        self._error_sum += np.sum(error,axis=self.axis)
        self._samples_num += tot

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num

# mean absolute error
class MAE(Error):
    def __init__(self,
        indexes=[1,2],
        reduce_all_dims=True,
        averaged_by_atoms=False,
        atom_aggregate='mean',
    ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self,y,y_pred):
        return np.abs(y.reshape(y_pred.shape) - y_pred)

# mean square error
class MSE(Error):
    def __init__(self,
        indexes=[1,2],
        reduce_all_dims=True,
        averaged_by_atoms=False,
        atom_aggregate='mean',
    ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self,y,y_pred):
        return np.square(y.reshape(y_pred.shape) - y_pred)

# mean norm error
class MNE(Error):
    def __init__(self,
        indexes=[1,2],
        reduce_all_dims=True,
        averaged_by_atoms=False,
        atom_aggregate='mean',
    ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self,y,y_pred):
        diff = y.reshape(y_pred.shape) - y_pred
        return np.linalg.norm(diff,axis=-1)

# root mean square error
class RMSE(Error):
    def __init__(self,
        indexes=[1,2],
        reduce_all_dims=True,
        averaged_by_atoms=False,
        atom_aggregate='mean',
    ):
        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self,y,y_pred):
        return np.square(y.reshape(y_pred.shape) - y_pred)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)

class MLoss(Metric):
    def __init__(self,index=0):
        super().__init__()
        self.clear()
        self._index = index

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):

        loss = self._convert_data(inputs[self._index])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError("Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num

class TransformerLR(LearningRateSchedule):
    def __init__(self, learning_rate=1.0, warmup_steps=4000, dimension=1):
        super().__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
        validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)

        self.learning_rate = learning_rate

        self.pow = P.Pow()
        self.warmup_steps = F.cast(warmup_steps,ms.float32)
        # self.warmup_scale = self.pow(F.cast(warmup_steps,ms.float32),-1.5)
        self.dimension = F.cast(dimension,ms.float32)
        # self.dim_scale = self.pow(F.cast(dimension,ms.float32),-0.5)

        self.min = P.Minimum()

    def construct(self, global_step):
        step_num = F.cast(global_step,ms.float32)
        warmup_scale = self.pow(self.warmup_steps,-1.5)
        dim_scale = self.pow(self.dimension,-0.5)
        lr1 = self.pow(step_num,-0.5)
        lr2 = step_num*warmup_scale
        lr_percent = dim_scale * self.min(lr1, lr2)
        return self.learning_rate * lr_percent