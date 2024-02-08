#!/usr/bin/env python
import os
from shutil import rmtree
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import subprocess
from openbabel import pybel


def get_dockingvina(target, exhaustiveness=1, num_cpu=8):
    docking_config = dict()

    if target == 'jak2':
        box_center = (114.758,65.496,11.345)
        box_size= (19.033,17.929,20.283)
    elif target == 'braf':
        box_center = (84.194,6.949,-7.081)
        box_size = (22.032,19.211,14.106)
    elif target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
    elif target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
    elif target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)

    docking_config['receptor_file']     = f'scorer/receptors/{target}.pdbqt'
    docking_config['vina_program']      = 'scorer/qvina02'
    docking_config['box_parameter']     = (box_center, box_size)
    docking_config['exhaustiveness']    = exhaustiveness
    docking_config['num_sub_proc']      = 10
    docking_config['num_cpu_dock']      = num_cpu
    docking_config['num_modes']         = 10
    docking_config['timeout_gen3d']     = 30
    docking_config['timeout_dock']      = 100

    return DockingVina(docking_config)


def make_docking_dir():
    for i in range(100):
        tmp_dir = f'tmp/tmp{i}'
        if not os.path.exists(tmp_dir):
            print(f'Docking tmp dir: {tmp_dir}')
            os.makedirs(tmp_dir)
            return tmp_dir
    raise ValueError('tmp/tmp0~99 are full. Please delete tmp dirs.')


class DockingVina(object):
    def __init__(self, docking_params):
        super(DockingVina, self).__init__()
        self.temp_dir = make_docking_dir()
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.vina_program = docking_params['vina_program']
        self.receptor_file = docking_params['receptor_file']
        box_parameter = docking_params['box_parameter']
        (self.box_center, self.box_size) = box_parameter
        self.exhaustiveness = docking_params['exhaustiveness']
        self.num_sub_proc = docking_params['num_sub_proc']
        self.num_cpu_dock = docking_params['num_cpu_dock']
        self.num_modes = docking_params['num_modes']
        self.timeout_gen3d = docking_params['timeout_gen3d']
        self.timeout_dock = docking_params['timeout_dock']

    def gen_3d(self, smi, ligand_mol_file):
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def docking_tmp(self, receptor_file, ligand_pdbqt_file, docking_pdbqt_file):
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                return_dict[idx] = 99.9
                continue
            if len(affinity_list) == 0:
                affinity_list.append(99.9)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator, args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess, args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list
    
    def __del__(self):
        if os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed.')
