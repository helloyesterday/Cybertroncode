# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Simulation with trained forcefield.
"""

from mindspore import context
from mindspore.nn import Adam

if __name__ == "__main__":

    simu_batch = 1
    simu_step = 10000
    simu_temp = 800
    simu_bias = 'bias' # nobias, metad, bias
    number = '1'
    
    import sys
    import os
    path = os.getenv('MINDSPONGE_HOME')
    if path:
        sys.path.insert(0, path)
    sys.path.append('../..')
    data_dir = './data'

    from sponge import Sponge
    from sponge import Molecule
    from sponge import WithEnergyCell
    from sponge import UpdaterMD
    from sponge import set_global_units
    from sponge.colvar import Distance
    from sponge.colvar import ColvarCombine

    import time
    import mindspore as ms
    import numpy as np
    from mindspore import Tensor
    from mindspore import load_checkpoint
    from cybertron.model import MolCT
    from cybertron.readout import AtomwiseReadout
    from cybertron.cybertron import CybertronFF

    from sponge.sampling import Metadynamics
    from sponge.callback import WriteH5MD, RunInfo
    from sponge.function import VelocityGenerator
    from sponge.control import VelocityVerlet, Langevin, BerendsenThermostat, LeapFrog
    from sponge.potential.bias import UpperWall, LowerWall

    import h5py

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="Set the backend.", default="GPU")
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.e)

    set_global_units('nm', 'kj/mol')

    atom_types = np.array([8,6,6,6,6,6,6,1,1,1,1,1,1,1,1],dtype=np.int32)
    atom_types = Tensor(atom_types,dtype=ms.int32)

    traj_file = data_dir + '/traj/PES_1-10000-800K-bias-NORMAL0.h5md'
    traj = h5py.File(traj_file)
    traj_array = np.array(traj['particles/trajectory/position/value'],dtype=float)
    rand_idx = np.random.choice(traj_array.shape[0], size=simu_batch, replace=False)
    coordinates = traj_array[rand_idx]
    rand_coord = Tensor(coordinates, dtype=ms.float32)
    system = Molecule(atomic_number=atom_types, coordinate=rand_coord)

    from sponge.data import read_yaml
    from cybertron import CybertronFF, load_checkpoint

    ckpt_file = data_dir + '/ckpt/cybertron-molct-trained-best.ckpt'
    config_file = data_dir + '/conf/configure_MolCT_trained.yaml'
    config = read_yaml(config_file)
    energy = CybertronFF(**config)
    load_checkpoint(ckpt_file, energy)

    scales = 4.677139 
    shifts = -538.89996
    energy.set_scaleshift(scale=scales, shift=shifts)


    seed = np.random.randint(1, 1000000)
    v_gen = VelocityGenerator(simu_temp,seed=seed)
    velocity = v_gen(system.coordinate.shape, system.atom_mass)

    d1 = Distance([[5,6]])
    d2 = Distance([[0,6]])
    d3 = Distance([[0,2]])
    d4 = Distance([[2,4]])
    d5 = Distance([[3,4]])
    d6 = Distance([[3,5]])

    combine = ColvarCombine(colvar=[d3,d6],weights=[-1.0, 1.0])
    metad = Metadynamics(
        colvar=combine,
        update_pace=20,
        height=4,
        sigma=0.01,
        grid_min=-0.85,
        grid_max=0.45,
        grid_bin=500,
        temperature=300,
        bias_factor=50,
        num_walker=simu_batch,
        share_parameter=False
    )
    CV_lwall = LowerWall(
        colvar = combine,
        boundary = [-0.86],
        energy_constant = 1000
    )
    CV_uwall = UpperWall(
        colvar = combine,
        boundary = [0.46],
        energy_constant = 1000
    )

    u_walls = UpperWall(
        colvar=[d1, d5, d6],
        boundary=[0.18, 0.2, 0.36],
        # depth=[1.0],
        energy_constant=[1000, 1000, 1000]
    )

    if simu_bias == 'nobias':
        sim = WithEnergyCell(system, potential=energy)
    elif simu_bias == 'metad':
        sim = WithEnergyCell(system, potential=energy, bias=[metad])
    elif simu_bias == 'bias':
        sim = WithEnergyCell(system, potential=energy, bias=[metad, u_walls, CV_lwall, CV_uwall])

    opt = UpdaterMD(
        system,
        integrator=LeapFrog(system),
        thermostat=Langevin(system, temperature=simu_temp, time_constant=0.5),
        # thermostat=BerendsenThermostat(system, temperature=simu_temp, time_constant=0.5),
        velocity=velocity,
        time_step=1e-3,
    )

    md = Sponge(sim, optimizer=opt, metrics={'d3': d3, 'd6': d6, 'd2': d2, 'd4': d4})

    save_path = data_dir + '/traj/'
    save_params = str(simu_batch) + '-' + str(simu_step) + '-' + str(simu_temp) + 'K-' + simu_bias + '-' +  number
    save_file_name = save_path + save_params + '.h5md'
    cb_h5md = WriteH5MD(system, save_file_name, save_freq=10, write_velocity=True, write_force=True)
    run_info = RunInfo(10)

    beg_time = time.time()
    md.run(simu_step, callbacks=[run_info,cb_h5md])
    
    end_time = time.time()

    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)
    print ("Run Time: %02d:%02d:%02d" % (h, m, s))

