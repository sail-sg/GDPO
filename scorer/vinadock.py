from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
import meeko
import warnings
import os
import sys
warnings.filterwarnings("ignore")

def vinadock(target,smiles_list):
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
    workdir = os.getcwd()
    print("vinadock os working dir",workdir)
    if "multirun" in workdir:
        home_prefix = "./../../../../"
    else:
        home_prefix = "./../../../"
    receptor = home_prefix+f'scorer/receptors/{target}.pdbqt'
    score_list = []
    for smiles in smiles_list:
        lig = Chem.MolFromSmiles(smiles)
        protonated_lig = Chem.AddHs(lig)
        status=AllChem.EmbedMolecule(protonated_lig)
        if status == -1:
            print("can't embed mol to 3d")
            score_list.append(-1)
            continue
        meeko_prep = meeko.MoleculePreparation()
        try:
            meeko_prep.prepare(protonated_lig)
            lig_pdbqt = meeko_prep.write_pdbqt_string()
        except:
            print("can't prepare lig_pdbqt")
            score_list.append(-1)
            continue
        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(receptor)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=box_center, box_size=box_size)
        # Score the current pose
        try:
            v.dock(exhaustiveness=16, n_poses=20)
            energy = v.score()
            # Minimized locally the current pose
            energy_minimized = v.optimize()
            score_list.append(-energy_minimized[0])
        except:
            print("can't get score")
            score_list.append(-1)
        
    return score_list

if __name__ == "__main__":
    print(vinadock("parp1",["Cc1noc(CCCN2CCC(=O)N(c3ccccc3O)CC2)n1","CC(=O)c1cc2ccc1c(=O)n2-c1ccc(C(=O)NC(C)C)s1"]))