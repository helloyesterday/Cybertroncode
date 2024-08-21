# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Load and evaluate forcefield model.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../mindscience/mindsponge')
sys.path.append('../..')
data_dir = '../../../mindscience/mindsponge/data'

test_dataset = data_dir + "/dataset/data_origin_testset_1024.npz"
testing = "not" # "trained" or others
if testing == "trained":
    ckpt = data_dir + "/ckpt/cybertron-molct-trained-best.ckpt"
    yaml = data_dir + "/conf/configure_MolCT_trained.yaml"
else:
    ckpt = data_dir + "/ckpt/cybertron-molct-best.ckpt"
    yaml = data_dir + "/conf/configure_MolCT.yaml"

import mindspore as ms
from mindspore import Tensor
from cybertron.model import MolCT
from cybertron.readout import AtomwiseReadout
from cybertron.cybertron import Cybertron
from sponge.data import read_yaml
from cybertron import load_checkpoint
from mindspore import context
from mindspore.ops import grad

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

config = read_yaml(yaml)
net = Cybertron(**config)
load_checkpoint(ckpt, net)

test_dataset = np.load(test_dataset)
scales = test_dataset['scale']
shifts = test_dataset['shift']
net.set_scaleshift(scale=scales, shift=shifts) 
print(scales, shifts)

coordinate = test_dataset['coordinate']
coordinate = Tensor(coordinate,  dtype=ms.float32)
real_E = test_dataset['label']
pred_E = net(coordinate).asnumpy()
pred_E = pred_E.reshape(-1)
real_E = real_E.flatten()

fig = plt.figure(figsize=(12.,12.))
pred_shift = pred_E - np.mean(pred_E)
real_shift = real_E - np.mean(real_E)

ax3 = fig.add_subplot(221)
ax3.scatter(real_shift / 4.184, pred_shift / 4.184 , s=1, label = 'predE')
ax3.plot(real_shift / 4.184, real_shift / 4.184, color='red', label = 'realE')
ax3.set_xlabel('B3LYP Energy (kcal/mol)')
ax3.set_ylabel('Predicted Energy (kcal/mol)')


grad_fn = grad(net)
pred_F = -grad_fn(coordinate).asnumpy()

real_F = test_dataset['force']

take_real_F = real_F.reshape(-1)
take_pred_F = pred_F.reshape(-1)

ax4 = fig.add_subplot(222)
ax4.scatter(take_real_F / 41.84, take_pred_F / 41.84, s=1, label='predF')
ax4.plot(take_real_F / 41.84, take_real_F / 41.84, color='red', label='realF')
ax4.set_xlabel('B3LYP Force (kcal/mol/$\AA$)')
ax4.set_ylabel('Predicted Force (kcal/mol/$\AA$)')

ax1 = fig.add_subplot(223)
ax1.scatter(real_shift / 4.184, (pred_shift-real_shift) /4.184, s=1)
ax1.plot(real_shift / 4.184, np.zeros_like(real_shift), color='red')
ax1.set_xlabel('B3LYP Energy (kcal/mol)')
ax1.set_ylabel('Energy Error (kcal/mol)')

ax2 = fig.add_subplot(224)
ax2.scatter(take_real_F / 41.84, (take_pred_F - take_real_F) / 41.84, s=1)
ax2.plot(take_real_F / 41.84, np.zeros_like(take_real_F), color='purple')
ax2.set_xlabel('B3LYP Force (kcal/mol/$\AA$)')
ax2.set_ylabel('Force Error (kcal/mol/$\AA$)')


#fig.savefig('revision/' + testing + 'ener_err.png',dpi=300, transparent=True)
#fig.savefig('revision/' + testing + 'force_err.png', dpi=300, transparent=True)
fig.show()


print(f"Energy MAE: {np.mean(np.abs(pred_shift - real_shift))/4.184}")
print(f"Force RMSE: {np.mean(np.linalg.norm(pred_F - real_F,axis=-1))/41.84}")
input()