import numpy as np
import torch
import re
import wandb
try:
    from rdkit import Chem
    print("Found rdkit, all good")
except ModuleNotFoundError as e:
    use_rdkit = False
    from warnings import warn
    warn("Didn't find rdkit, this will fail")
    assert use_rdkit, "Didn't find rdkit"
allowed_bonds = {"H": [1,2], "Sr":[1,2],"Nd":[2,3,4],"In":[1,2,3,5],"Ba":[1,2],"Cd":[1,2],"Li": 1, "B": [3,4], "C": [1,2,3,4], "N": [1,2,3,4], "O": [1,2,3], "F": 1, "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": [3, 5],
                 "S": [1,2,3,4,6], "Cl": 1, "K": 1, "Ca": 2, "Ti": 4, "V": 6, "Cr": [3, 6], "Mn": [2, 3, 4, 5, 6, 7, 1], "Fe": [2, 3, 6], "Co": 2, "Ni": [2, 3], "Cu": [1, 2], "Zn": 2, "Ga": 3, "Ge": 4,
                 "As": [3, 5], "Se": [2, 3, 4], "Br": 1, "Zr": [2, 3, 4], "Mo": [2, 3, 4, 5, 6], "Ru": [2, 3, 4, 8], "Rh": [2, 4, 6], "Pd": [2, 3, 4], "Ag": [0,1], "Sn": [2, 4], "Sb": [3, 5], "Te": [2, 6], "I": 1, "Cs": 1, "Gd": [3,4],
                 "Tb": [3, 4], "Ho": 3, "W": 6, "Re": [3, 4, 6, 7], "Ir": [2, 6], "Pt": [2, 4], "Au": [1, 3], "Hg": [1, 2], "Tl": [1, 3, 4], "Pb": 4, "Bi": [3, 5], "Ac": 1, "U": 10}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
from collections import defaultdict
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []

        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types,
                                 self.dataset_info.atom_decoder)
            try:
                smiles = mol2smiles(mol)
            except:
                smiles = None
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder)
            try:
                smiles = mol2smiles(mol)
            except:
                smiles = None
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, num_components, all_smiles = self.compute_validity(
            generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(
            f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(
            f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(
            generated)
        print(
            f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(
                f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(
                    f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []
        return ([validity, relaxed_validity, uniqueness, novelty], unique,
                dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), all_smiles)


class BasicMolecularMetricslist(object):
    def __init__(self, dataset_info, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        valid_list = []
        reward_list = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types,
                                 self.dataset_info.atom_decoder)
            try:
                smiles = mol2smiles(mol)
            except:
                smiles = None
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                    valid_list.append(1)
                    reward_list.append(1)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                    valid_list.append(0)
                    reward_list.append(0.)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
                    valid_list.append(0)
                    reward_list.append(0.)
            else:
                all_smiles.append(None)
                valid_list.append(0)
                reward_list.append(0)
        freq_list = [0]*len(generated)
        freq_dict = defaultdict(int)
        for smile in all_smiles:
            if smile is not None:
                freq_dict[smile]+=1
        for idx,smile in enumerate(all_smiles):
            if smile is not None:
                freq_list[idx] = freq_dict[smile]
        
        uniq,uniq_r = self.compute_uniqueness(valid)

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles,valid_list,uniq,freq_list,reward_list 
    def get_smiles(self,generated):
        smile_list = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types,
                                 self.dataset_info.atom_decoder)
            try:
                smiles = mol2smiles(mol)
            except:
                smiles = None
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    smile_list.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    smile_list.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    smile_list.append(None)
            else:
                smile_list.append(None)
        freq_list = [0]*len(generated)
        freq_dict = defaultdict(int)
        for smile in smile_list:
            if smile is not None:
                freq_dict[smile]+=1
        for idx,smile in enumerate(smile_list):
            if smile is not None:
                freq_list[idx] = freq_dict[smile]
        valid = [x for x in smile_list if x is not None]
        valid_r = len(valid)/len(smile_list)
        uniq,uniq_r = self.compute_uniqueness(valid)
        return smile_list,valid_r,uniq,uniq_r,freq_list

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / (len(valid)+1e-8)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol,
                                      key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, num_components, all_smiles,valid_list,uniq,freq_list,reward_list = self.compute_validity(
            generated)
        # nc_mu = num_components.mean() if len(num_components) > 0 else 0
        # nc_min = num_components.min() if len(num_components) > 0 else 0
        # nc_max = num_components.max() if len(num_components) > 0 else 0
        # print(
        #     f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        # print(
        #     f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        # relaxed_valid, relaxed_validity = self.compute_relaxed_validity(
        #     generated)
        # print(
        #     f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        # if relaxed_validity > 0:
        #     unique, uniqueness = self.compute_uniqueness(relaxed_valid)
        #     print(
        #         f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%")

        #     if self.dataset_smiles_list is not None:
        #         _, novelty = self.compute_novelty(unique)
        #         print(
        #             f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
        #     else:
        #         novelty = -1.0
        # else:
        #     novelty = -1.0
        #     uniqueness = 0.0
        #     unique = []
        return valid_list,uniq,freq_list,reward_list

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()])
    return mol


def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)

    for i, bond in enumerate(all_bonds):
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                      bond_dict[edge_types[bond[0], bond[1]].item()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append(
                    (b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm = [(s, len(s)) for s in sm.split('.')]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


# if __name__ == '__main__':
#     smiles_mol = 'C1CCC1'
#     print("Smiles mol %s" % smiles_mol)
#     chem_mol = Chem.MolFromSmiles(smiles_mol)
#     block_mol = Chem.MolToMolBlock(chem_mol)
#     print("Block mol:")
#     print(block_mol)

use_rdkit = True


def check_stability(atom_types, edge_types, dataset_info, debug=False):
    atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" %
                  (atom_decoder[atom_type], atom_n_bond))
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def compute_molecular_metrics(molecule_list, train_smiles, dataset_info):
    """ molecule_list: (dict) """

    if not dataset_info.remove_h:
        print(f'Analyzing molecule stability...')

        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        n_molecules = len(molecule_list)

        for i, mol in enumerate(molecule_list):
            atom_types, edge_types = mol

            validity_results = check_stability(
                atom_types, edge_types, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])

        # Validity
        fraction_mol_stable = molecule_stable / float(n_molecules)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        validity_dict = {'mol_stable': fraction_mol_stable,
                         'atm_stable': fraction_atm_stable}
        if wandb.run:
            wandb.log(validity_dict)
    else:
        validity_dict = {'mol_stable': -1, 'atm_stable': -1}

    metrics = BasicMolecularMetrics(dataset_info, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]
    if wandb.run:
        nc = rdkit_metrics[-2]
        dic = {'Validity': rdkit_metrics[0][0], 'Relaxed Validity': rdkit_metrics[0][1],
               'Uniqueness': rdkit_metrics[0][2], 'Novelty': rdkit_metrics[0][3],
               'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu']}
        wandb.log(dic)

    return validity_dict, rdkit_metrics, all_smiles

def compute_molecular_metrics_list(molecule_list,dataset_info):
    """ molecule_list: (dict) """

    print(f'Analyzing molecule stability...')

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = len(molecule_list)

    metrics = BasicMolecularMetricslist(dataset_info, dataset_info.train_smiles)
    valid_list,uniq,freq_list,reward_list = metrics.evaluate(molecule_list)
    return valid_list,uniq,freq_list,reward_list

def gen_smile_list(molecule_list,dataset_info):
    metrics = BasicMolecularMetricslist(dataset_info, dataset_info.train_smiles)
    smile_list,valid_r,uniq,uniq_r,freq_list = metrics.get_smiles(molecule_list)
    return smile_list,valid_r,uniq,uniq_r,freq_list

