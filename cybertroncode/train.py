import os
import numpy as np
import mindspore as ms
from shutil import copyfile
from collections import deque
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from mindspore.nn.metrics import Metric
from mindspore.train._utils import _make_directory
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore._checkparam import Validator as validator

_cur_dir = os.getcwd()

__all__ = [
    "DatasetWhitening",
    "OutputScaleShift",
    "WithForceLossCell",
    "WithLabelLossCell",
    "WithForceEvalCell",
    "WithLabelEvalCell",
    "EvalMonitor",
    "MAE",
    "MSE",
    "MAEAveragedByAtoms",
    "MLoss",
    "TransformerLR",
    ]

class DatasetWhitening(nn.Cell):
    def __init__(self,scale=1.,shift=0.,references=None,mask=None,axis=-2,):
        super().__init__()

        self.scale = scale
        self.shift = shift

        self.references = references
        self.axis = axis
        self.mask = mask

        self.scaled_by_atoms = self.references is not None
        self.scaled_by_graph = not self.scaled_by_atoms

        self.mixed_scale_type = self.mask is not None
        if self.mixed_scale_type:
            if self.references is None:
                raise TypeError('references must be given at mixed scale type')
            self.scaled_by_graph = True

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, label, types=None):

        graph_scale = None
        if self.scaled_by_graph:
            graph_scale = (label - self.shift) / self.scale
        
        atoms_scale = None
        if self.scaled_by_atoms:
            atom_num = F.cast(types>0,label.dtype)
            atom_num = self.keep_sum(atom_num,-1)

            ref = F.gather(self.references,types,0)
            ref = self.reduce_sum(ref,-2)

            atoms_scale = (label - ref - self.shift * atom_num) / self.scale
        
        if self.mixed_scale_type:
            mask = self.mask * F.ones_like(label)
            return F.select(mask,atoms_scale,graph_scale)
        else:
            return atoms_scale if self.scaled_by_atoms else graph_scale

class OutputScaleShift(nn.Cell):
    def __init__(self,scale=1.,shift=0.,references=None,mask=None,axis=-2):
        super().__init__()

        self.scale = scale
        self.shift = shift

        self.references = references
        self.axis = axis
        self.mask = mask

        self.scaled_by_atoms = self.references is not None
        self.scaled_by_graph = not self.scaled_by_atoms

        self.mixed_scale_type = self.mask is not None
        if self.mixed_scale_type:
            if self.references is None:
                raise TypeError('references must be given at mixed scale type')
            self.scaled_by_graph = True

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(keep_dims=True)

    def construct(self, outputs, types=None):

        graph_scale = None
        if self.scaled_by_graph:
            graph_scale = outputs * self.scale + self.shift
        
        atoms_scale = None
        if self.scaled_by_atoms:
            atom_num = F.cast(types>0,outputs.dtype)
            atom_num = self.keep_sum(atom_num,-1)

            ref = F.gather(self.references,types,0)
            ref = self.reduce_sum(ref,-2)

            atoms_scale = outputs * self.scale + self.shift * atom_num + ref
        
        if self.mixed_scale_type:
            mask = self.mask * F.ones_like(outputs)
            return F.select(mask,atoms_scale,graph_scale)
        else:
            return atoms_scale if self.scaled_by_atoms else graph_scale

class WithForceLossCell(nn.Cell):
    def __init__(self,
        datatypes,
        backbone,
        energy_fn,
        force_fn,
        ratio_energy=0.01,
        ratio_force=0.99,
        # with_penalty=False,
    ):
        super().__init__(auto_prefix=False)

        if not isinstance(datatypes,str):
            raise TypeError('Type of "datatypes" must be str')

        fulltypes = 'RZANnBbLlFE'
        for datatype in datatypes:
            if fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in fulltypes:
            num = datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype + '" in datatype "' + datatypes + '".')

        self.R = datatypes.find('R') # positions
        self.Z = datatypes.find('Z') # node_types
        self.A = datatypes.find('A') # atom_types
        self.N = datatypes.find('N') # neighbors
        self.n = datatypes.find('n') # neighbor_mask
        self.B = datatypes.find('B') # bonds
        self.b = datatypes.find('b') # bond_mask
        self.L = datatypes.find('L') # far_neighbors
        self.l = datatypes.find('l') # far_mask
        self.F = datatypes.find('F') # force
        self.E = datatypes.find('E') # energy

        if self.F < 0:
            raise TypeError('The datatype "F" must be included in WithForceLossCell!')

        if self.E < 0:
            raise TypeError('The datatype "E" must be included in WithForceLossCell!')

        self._backbone = backbone
        self.force_fn = force_fn
        self.energy_fn = energy_fn
        self.ratio_energy = ratio_energy
        self.ratio_force = ratio_force
        self.grad_op = C.GradOperation(get_all=False)
        # self.with_penalty = with_penalty

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        node_types = inputs[self.Z]
        atom_types = inputs[self.A]
        neighbors = inputs[self.N]
        neighbor_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]
        far_neighbors = inputs[self.L]
        far_mask = inputs[self.l]

        out = self._backbone(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )
        fout = -1 * self.grad_op(self._backbone)(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        forces = inputs[self.F]
        energy = inputs[self.E]
        
        loss_force = self.force_fn(fout,forces) * self.ratio_force
        loss_energy = self.energy_fn(out,energy) * self.ratio_energy
        return loss_energy + loss_force

    @property
    def backbone_network(self):
        return self._backbone

class WithLabelLossCell(nn.Cell):
    def __init__(self,
        datatypes,
        backbone,
        loss_fn,
        do_whitening=False,
        scale=1,
        shift=0,
        references=None,
        mask=None,
        # with_penalty=False,
    ):
        super().__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        # self.with_penalty = with_penalty

        self.do_whitening = do_whitening
        if do_whitening:
            self.whitening = DatasetWhitening(
                scale=scale,
                shift=shift,
                references=references,
                mask=mask
            )
        else:
            self.whitening = None

        if not isinstance(datatypes,str):
            raise TypeError('Type of "datatypes" must be str')

        fulltypes = 'RZANnBbLlE'
        for datatype in datatypes:
            if fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in fulltypes:
            num = datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype + '" in datatype "' + datatypes + '".')

        self.R = datatypes.find('R') # positions
        self.Z = datatypes.find('Z') # node_types
        self.A = datatypes.find('A') # atom_types
        self.N = datatypes.find('N') # neighbors
        self.n = datatypes.find('n') # neighbor_mask
        self.B = datatypes.find('B') # bonds
        self.b = datatypes.find('b') # bond_mask
        self.L = datatypes.find('L') # far_neighbors
        self.l = datatypes.find('l') # far_mask
        self.E = datatypes.find('E') # label

        if self.E < 0:
            raise TypeError('The datatype "E" must be included in WithLabelLossCell!')

    def construct(self, *inputs):

        inputs = inputs + (None,)

        positions = inputs[self.R]
        node_types = inputs[self.Z]
        atom_types = inputs[self.A]
        neighbors = inputs[self.N]
        neighbor_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]
        far_neighbors = inputs[self.L]
        far_mask = inputs[self.l]

        out = self._backbone(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        label = inputs[self.E]
        types = atom_types
        if types is None:
            types = node_types

        if self.do_whitening:
            label = self.whitening(label,types)

        return self._loss_fn(out,label)

class WithForceEvalCell(nn.Cell):
    def __init__(self,
        datatypes,
        network,
        energy_fn=None,
        force_fn=None,
        add_cast_fp32=False
    ):
        super().__init__(auto_prefix=False)

        if not isinstance(datatypes,str):
            raise TypeError('Type of "datatypes" must be str')

        fulltypes = 'RZANnBbLlFE'
        for datatype in datatypes:
            if fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in fulltypes:
            num = datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype + '" in datatype "' + datatypes + '".')

        self.R = datatypes.find('R') # positions
        self.Z = datatypes.find('Z') # node_types
        self.A = datatypes.find('A') # atom_types
        self.N = datatypes.find('N') # neighbors
        self.n = datatypes.find('n') # neighbor_mask
        self.B = datatypes.find('B') # bonds
        self.b = datatypes.find('b') # bond_mask
        self.L = datatypes.find('L') # far_neighbors
        self.l = datatypes.find('l') # far_mask
        self.F = datatypes.find('F') # forces
        self.E = datatypes.find('E') # energy

        if self.F < 0:
            raise TypeError('The datatype "F" must be included in WithForceEvalCell!')

        if self.E < 0:
            raise TypeError('The datatype "E" must be included in WithForceEvalCell!')

        self._network = network
        self._energy_fn = energy_fn
        self._force_fn = force_fn
        self.add_cast_fp32 = add_cast_fp32

        self.grad_op = C.GradOperation(get_all=False)

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        node_types = inputs[self.Z]
        atom_types = inputs[self.A]
        neighbors = inputs[self.N]
        neighbor_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]
        far_neighbors = inputs[self.L]
        far_mask = inputs[self.l]

        outputs = self._network(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )
        foutputs = -1 * self.grad_op(self._network)(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )

        forces = inputs[self.F]
        energy = inputs[self.E]

        if self.add_cast_fp32:
            forces = F.mixed_precision_cast(ms.float32, forces)
            energy = F.mixed_precision_cast(ms.float32, energy)
            outputs = F.cast(outputs, ms.float32)

        if self._energy_fn is None:
            eloss = 0
        else:
            eloss = self._energy_fn(outputs, energy)

        if self._force_fn is None:
            floss = 0
        else:
            floss = self._force_fn(foutputs, forces)
        
        return eloss, floss, outputs, energy, foutputs, forces

class WithLabelEvalCell(nn.Cell):
    def __init__(self,
        datatypes,
        network,
        loss_fn=None,
        add_cast_fp32=False,
        do_scaleshift=False,
        scale=1,
        shift=0,
        references=None,
        mask=None,
    ):
        super().__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = add_cast_fp32
        self.reducesum = P.ReduceSum(keep_dims=True)

        self.do_scaleshift = do_scaleshift
        self.scaleshift = None
        self.whitening = None
        if do_scaleshift:
            self.scaleshift = OutputScaleShift(
                scale=scale,
                shift=shift,
                references=references,
                mask=mask
            )
            if loss_fn is not None:
                self.whitening = DatasetWhitening(
                    scale=scale,
                    shift=shift,
                    references=references,
                    mask=mask
                )

        if not isinstance(datatypes,str):
            raise TypeError('Type of "datatypes" must be str')

        fulltypes = 'RZANnBbLlE'
        for datatype in datatypes:
            if fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in fulltypes:
            num = datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype + '" in datatype "' + datatypes + '".')

        self.R = datatypes.find('R') # positions
        self.Z = datatypes.find('Z') # node_types
        self.A = datatypes.find('A') # atom_types
        self.N = datatypes.find('N') # neighbors
        self.n = datatypes.find('n') # neighbor_mask
        self.B = datatypes.find('B') # bonds
        self.b = datatypes.find('b') # bond_mask
        self.L = datatypes.find('L') # far_neighbors
        self.l = datatypes.find('l') # far_mask
        self.E = datatypes.find('E') # label

        if self.E < 0:
            raise TypeError('The datatype "E" must be included in WithLabelEvalCell!')

    def construct(self, *inputs):
        inputs = inputs + (None,)

        positions = inputs[self.R]
        node_types = inputs[self.Z]
        atom_types = inputs[self.A]
        neighbors = inputs[self.N]
        neighbor_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]
        far_neighbors = inputs[self.L]
        far_mask = inputs[self.l]

        outputs = self._network(
            positions,
            node_types,
            atom_types,
            neighbors,
            neighbor_mask,
            bonds,
            bond_mask,
            far_neighbors,
            far_mask,
        )
        label = inputs[self.E]
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(ms.float32, label)
            outputs = F.cast(outputs, ms.float32)

        types = atom_types
        if types is None:
            types = node_types

        loss = 0
        if self._loss_fn is not None:
            _label = label
            if self.do_scaleshift:
                _label = self.whitening(label,types)
            loss = self._loss_fn(outputs, _label)

        if self.do_scaleshift:
            outputs = self.scaleshift(outputs,types)

        dataref = positions if positions is not None else outputs
        atom_num = F.cast(types>0,dataref.dtype)
        atom_num = self.reducesum(atom_num,-1)

        return loss, outputs, label, atom_num

class EvalMonitor(Callback):
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
        self._ckptdata = name + '-best-ckpt.data'

        self.num_ckpt = 1
        self.best_value = 5e4
        self.best_ckpt_metrics = best_ckpt_metrics

        self.last_loss = 0
        self.record = []

        self.output_title = True
        filename = os.path.join(self._directory, self._filename)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(filename)

    def _write_cpkt_file(self,filename,info,network):
        ckptfile = os.path.join(self._directory, filename + '.ckpt')
        ckptbck = os.path.join(self._directory, filename + '.bck.ckpt')
        ckptdata = os.path.join(self._directory, self._ckptdata)

        if os.path.exists(ckptfile):
            os.rename(ckptfile,ckptbck)
        save_checkpoint(network,ckptfile)
        with open(ckptdata, "a") as f:
            f.write(info + os.linesep)

    def _output_data(self,cb_params):
        cur_epoch = cb_params.cur_epoch_num
        opt = cb_params.optimizer
        if opt is None:
            opt = cb_params.train_network.optimizer
        global_step = opt.global_step
        if not isinstance(global_step,int):
            global_step = global_step.asnumpy()[0]
        global_step = F.cast(global_step,ms.int32)

        if self.avg_steps > 0:
            mov_avg = sum(self.loss_record) / sum(self.train_num)
        else:
            mov_avg = self.loss_record / self.train_num
        
        title = "#! FIELDS step"
        info = 'Epoch: ' + str(cur_epoch) + ', Step: ' + str(global_step)
        outdata = '{:>10d}'.format(global_step.asnumpy())

        lr = opt.learning_rate
        if opt.dynamic_lr:
            if opt.is_group_lr:
                lr = ()
                for learning_rate in opt.learning_rate:
                    current_dynamic_lr = learning_rate(global_step-1)
                    lr += (current_dynamic_lr,)
            else:
                lr = opt.learning_rate(global_step-1)

        title += ' learning_rate'
        info += ', Learning_rate: ' + str(lr)
        outdata += '{:>15e}'.format(lr.asnumpy())

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

                if type(v) is np.ndarray and len(v) > 1:
                    for i in range(len(v)):
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
                        self._write_cpkt_file(self._ckptfile,info,cb_params.train_network)
                        source_ckpt = os.path.join(self._directory, self._ckptfile + '.ckpt')
                        for i in range(len(vnow)):
                            if output_ckpt[i]:
                                dest_ckpt = os.path.join(self._directory, self._ckptfile + '-' + str(i) + '.ckpt')
                                bck_ckpt = os.path.join(self._directory, self._ckptfile + '-' + str(i) + '.bck.ckpt')
                                if os.path.exists(dest_ckpt):
                                    os.rename(dest_ckpt,bck_ckpt)
                                copyfile(source_ckpt,dest_ckpt)
                        self.best_value = np.minimum(vnow,self.best_value)
                else:
                    if vnow < self.best_value:
                        self._write_cpkt_file(self._ckptfile,info,cb_params.train_network)
                        self.best_value = vnow

        print(info, flush=True)
        filename = os.path.join(self._directory, self._filename)
        if self.output_title:
            with open(filename, "a") as f:
                f.write(title + os.linesep)
            self.output_title = False
        with open(filename, "a") as f:
            f.write(outdata + os.linesep)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
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

class MAE(Metric):
    def __init__(self,indexes=[2,3],reduce_all_dims=True):
        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._abs_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        abs_error = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += np.average(abs_error,axis=self.axis) * y.shape[0]
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num

class MAEAveragedByAtoms(Metric):
    def __init__(self,indexes=[1,2,3],reduce_all_dims=True):
        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._abs_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        n = self._convert_data(inputs[self._indexes[2]])

        abs_error = np.abs(y.reshape(y_pred.shape) - y_pred) / n
        self._abs_error_sum += np.average(abs_error,axis=self.axis) * y.shape[0]
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num

class MSE(Metric):
    def __init__(self,indexes=[4,5],reduce_all_dims=True):
        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._abs_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        error = y.reshape(y_pred.shape) - y_pred
        error = np.linalg.norm(error,axis=-1)
        self._abs_error_sum += np.average(error,axis=self.axis) * y.shape[0]
        self._samples_num += y.shape[0]

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num


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
        self.warmup_scale = self.pow(F.cast(warmup_steps,ms.float32),-1.5)
        self.dim_scale = self.pow(F.cast(dimension,ms.float32),-0.5)

        self.min = P.Minimum()

    def construct(self, global_step):
        step_num = F.cast(global_step,ms.float32)
        lr_percent = self.dim_scale * self.min(self.pow(step_num,-0.5), step_num*self.warmup_scale)
        return self.learning_rate * lr_percent