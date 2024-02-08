import utils
import torch
import os
import sys
from rdkit import Chem
o_path = os.getcwd()
sys.path.append(o_path)
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]


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
# OGB source code


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(
            bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(
            str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(
            bond.GetIsConjugated()),
    ]
    return bond_feature


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(
            allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        safe_index(allowable_features['possible_chirality_list'], str(
            atom.GetChiralTag())),
        # safe_index(
        #     allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(
            allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(
            allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(
            allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(
            atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(
            atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)
        self.atom_decoder = dataset_infos.atom_decoder
        self.name = dataset_infos.name
        if self.name == "OGBK":
            self.ogb = OGBFeature(self.atom_decoder)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(noisy_data)                    # (bs, 1)
        extra_edge_attr = torch.zeros(
            (*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t'])
        X = torch.cat((charge, valency), dim=-1)
        E = extra_edge_attr
        if self.name == "OGBK":
            ex_x, ex_E = self.ogb(noisy_data)
            X = torch.cat([X, ex_x], dim=-1)
            E = ex_E
        return utils.PlaceHolder(X=X, E=E, y=weight)


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        bond_orders = torch.tensor(
            [0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        weighted_E = noisy_data['E_t'] * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(
            self.valencies, device=noisy_data['X_t'].device).reshape(1, 1, -1)
        # print(noisy_data['X_t'].shape,valencies.shape)
        X = noisy_data['X_t'] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(noisy_data['X_t'])


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        orders = torch.tensor(
            [0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        E = noisy_data['E_t'] * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(noisy_data['X_t'])


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.tensor(list(atom_weights.values()))

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data['X_t'], dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list[X]            # (bs, n)
        # (bs, 1)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(noisy_data['X_t']) / self.max_weight


class OGBFeature:
    def __init__(self, atom_decoder):
        self.atom_decoder = atom_decoder

    def __call__(self, noisy_data):
        edge_type = noisy_data['E_t'].argmax(dim=-1)
        atom_type = torch.argmax(noisy_data['X_t'], dim=-1)
        mol_list = []
        for idx, edges in enumerate(edge_type):
            atoms = atom_type[idx]
            mol_list.append(build_molecule(atoms, edges, self.atom_decoder))
        x_list = []
        e_list = []
        for mol in mol_list:
            atom_num = mol.GetNumAtoms()
            bond_num = mol.GetNumBonds()
            E = torch.zeros((atom_num, atom_num, 2))
            x = []
            for idx in range(atom_num):
                atom = mol.GetAtomWithIdx(idx)
                atom_feature = atom_to_feature_vector(atom)[1:]
                x.append(atom_feature)
            x = torch.Tensor(x)
            for idx in range(bond_num):
                bond = mol.GetBondWithIdx(idx)
                start_id, end_id = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                e = bond_to_feature_vector(bond)[1:]
                E[start_id, end_id, :] = torch.Tensor(e)
                E[end_id, start_id, :] = torch.Tensor(e)
            x_list.append(x)
            e_list.append(E)
        return torch.stack(x_list), torch.stack(e_list)
