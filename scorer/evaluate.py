from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import numpy as np
import pandas as pd
import json
import torch
import networkx as nx
import time
import re
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scorer.scorer import get_scores
from scorer.vinadock import vinadock
import warnings
import random
warnings.filterwarnings("ignore")

def get_novelty_in_df(df,sr=1.,train_fps=None):
    # train_smiles, _ = load_smiles()
    # train_smiles = random.sample(train_smiles,int(sr*len(train_smiles)))
    # train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    train_fps = random.sample(train_fps,int(sr*len(train_fps)))
    
    if 'sim' not in df.keys():
        gen_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['mol']]
        # train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]
        max_sims = []
        for i in range(len(gen_fps)):
            sims = DataStructs.BulkTanimotoSimilarity(gen_fps[i], train_fps)
            max_sims.append(max(sims))
        df['sim'] = max_sims
def gen_score_list(protein,smiles,train_fps=None,weight_list=None):
    df = pd.DataFrame()
    if len(smiles)==0:
        return []
    num_mols = len(smiles)

    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    validity = len(df) / (num_mols+1e-8)
    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df,0.5,train_fps)
    
    df[protein] = vinadock(protein,smiles)
    neg_prop =(df[protein]==-1).tolist()
    df = df[~df[protein].isin([-1])]
    dsscore = np.clip(df[protein],0,20)/10
    novelscore=1-df["sim"]
    df['qed'] = get_scores('qed', df['mol'])
    qedscore = np.array(df["qed"])
    df['sa'] = get_scores('sa', df['mol'])
    sascore = np.array(df["sa"])
    if weight_list is None:
        score_list = 0.1*qedscore+0.1*sascore+0.4*novelscore+0.4*dsscore
    else:
        score_list = weight_list[0]*qedscore+weight_list[1]*sascore+weight_list[2]*novelscore+weight_list[3]*dsscore

    valid_score_list = score_list.tolist()
    score_list = [-1]*len(neg_prop)
    pos = 0
    for idx,value in enumerate(neg_prop):
        if value:
            continue
        else:
            score_list[idx] = valid_score_list[pos]
            pos+=1
    return score_list

def gen_score_disc_list(protein,smiles,thres=0.9,train_fps=None,weight_list=None):
    df = pd.DataFrame()
    if len(smiles)==0:
        return []
    num_mols = len(smiles)

    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    validity = len(df) / (num_mols+1e-8)
    uniqueness = len(set(df['smiles'])) / len(df)
    
    # novelscore=1-df["sim"]
    df[protein] = vinadock(protein,smiles)
    neg_prop =(df[protein]==-1).tolist()
    df = df[~df[protein].isin([-1])]
    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    dsscore = (np.array(df[protein])>(thres*hit_thr)).astype("float")*np.array(df[protein])/10
    get_novelty_in_df(df,0.5,train_fps)
    novelscore = 1-df["sim"]
    novelscore = (novelscore>=0.6).astype("float")*novelscore
    df['qed'] = get_scores('qed', df['mol'])
    qedscore = np.array(df["qed"])
    df['sa'] = get_scores('sa', df['mol'])
    sascore = np.array(df["sa"])
    if weight_list is None:
        score_list = 0.1*qedscore+0.1*sascore+0.3*novelscore+0.5*dsscore
    else:
        score_list = weight_list[0]*qedscore+weight_list[1]*sascore+weight_list[2]*novelscore+weight_list[3]*dsscore
    valid_score_list = score_list.tolist()
    score_list = [-1]*len(neg_prop)
    pos = 0
    for idx,value in enumerate(neg_prop):
        if value:
            continue
        else:
            score_list[idx] = valid_score_list[pos]
            pos+=1
    return score_list
def evaluate(protein, smiles, mols=None,train_fps=None):
    df = pd.DataFrame()
    num_mols = len(smiles)
    if num_mols==0:
        return {'validity': 0, 'uniqueness': 0,
            'novelty': 0, 'top_ds': 0, 'hit': 0,"avgscore":0,"avgds":0,"avgqed":0,"avgsa":0}
    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    validity = len(df) / num_mols

    if mols is None:
        df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        df['mol'] = mols

    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df,1.,train_fps)
    novelty = len(df[df['sim'] < 0.4]) / len(df)

    df = df.drop_duplicates(subset=['smiles'])
    before_num = len(df)
    df[protein] = vinadock(protein,df["smiles"].tolist())
    df = df[~df[protein].isin([-1])]
    after_num = len(df)
    num_mols = num_mols-(before_num-after_num)

    df['qed'] = get_scores('qed', df['mol'])

    df['sa'] = get_scores('sa', df['mol'])
    avgscore = ((df[protein]/10)*df["qed"]*df["sa"]).mean()
    avgds = (df[protein]/10).mean()
    avgqed = df["qed"].mean()
    avgsa = df["sa"].mean()
    del df['mol']
    # df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    else: raise ValueError('Wrong target protein')
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.4]
    df = df.sort_values(by=[protein], ascending=False)
    num_top5 = int(num_mols * 0.05)
    
    top_ds = df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std()
    hit = len(df[df[protein] > hit_thr]) / (num_mols+1e-6)
    
    return {'validity': validity, 'uniqueness': uniqueness,
            'novelty': novelty, 'top_ds': top_ds, 'hit': hit,"avgscore":avgscore,"avgds":avgds,"avgqed":avgqed,"avgsa":avgsa}

def evaluatelist(protein, smiles, mols=None,train_fps=None):
    df = pd.DataFrame()
    num_mols = len(smiles)
    if num_mols==0:
        return {'validity': 0, 'uniqueness': 0,
            'novelty': 0, 'top_ds': 0, 'hit': 0,"avgscore":0,"avgds":0,"avgqed":0,"avgsa":0}
    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    # validity = len(df) / num_mols

    if mols is None:
        df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        df['mol'] = mols

    get_novelty_in_df(df,1.,train_fps)

    df[protein] = vinadock(protein,df["smiles"].tolist())
    
    df['qed'] = get_scores('qed', df['mol'])

    df['sa'] = get_scores('sa', df['mol'])
    
    del df['mol']
    # df.to_csv(f'{csv_dir}.csv', index=False)

    return df

def gen_score_disc_listmose(protein,smiles,thres=0.9,train_fps=None):
    df = pd.DataFrame()
    if len(smiles)==0:
        return []
    num_mols = len(smiles)

    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    validity = len(df) / (num_mols+1e-8)
    uniqueness = len(set(df['smiles'])) / len(df)
    
    # novelscore=1-df["sim"]
    df[protein] = vinadock(protein,smiles)
    neg_prop =(df[protein]==-1).tolist()
    df = df[~df[protein].isin([-1])]
    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    dsscore = (np.array(df[protein])>(thres*hit_thr)).astype("float")*np.array(df[protein])/10
    get_novelty_in_df(df,0.5,train_fps)
    novelscore = 1-df["sim"]
    novelscore = (novelscore>=0.4).astype("float")*novelscore
    df['qed'] = get_scores('qed', df['mol'])
    qedscore = np.array(df["qed"])
    df['sa'] = get_scores('sa', df['mol'])
    sascore = np.array(df["sa"])
    score_list = 0.1*qedscore+0.1*sascore+0.3*novelscore+0.5*dsscore
    valid_score_list = score_list.tolist()
    score_list = [-1]*len(neg_prop)
    pos = 0
    for idx,value in enumerate(neg_prop):
        if value:
            continue
        else:
            score_list[idx] = valid_score_list[pos]
            pos+=1
    return score_list

# def gen_score_listmose(protein,smiles,train_fps=None):
#     df = pd.DataFrame()
#     if len(smiles)==0:
#         return []
#     num_mols = len(smiles)

#     # remove empty molecules
#     while True:
#         if '' in smiles:
#             idx = smiles.index('')
#             del smiles[idx]
#             if mols is not None:
#                 del mols[idx]
#         else:
#             break
#     df['smiles'] = smiles
#     df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
#     validity = len(df) / (num_mols+1e-8)
#     uniqueness = len(set(df['smiles'])) / len(df)
    
#     # novelscore=1-df["sim"]
#     df[protein] = vinadock(protein,smiles)
#     neg_prop =(df[protein]==-1).tolist()
#     df = df[~df[protein].isin([-1])]
#     if protein == 'parp1': hit_thr = 10.
#     elif protein == 'fa7': hit_thr = 8.5
#     elif protein == '5ht1b': hit_thr = 8.7845
#     elif protein == 'jak2': hit_thr = 9.1
#     elif protein == 'braf': hit_thr = 10.3
#     dsscore = np.array(df[protein])/10
#     get_novelty_in_df(df,0.5,train_fps)
#     novelscore = 1-df["sim"]
#     # novelscore = (novelscore>=0.4).astype("float")*novelscore
#     df['qed'] = get_scores('qed', df['mol'])
#     qedscore = np.array(df["qed"])
#     df['sa'] = get_scores('sa', df['mol'])
#     sascore = np.array(df["sa"])
#     score_list = 0.1*qedscore+0.1*sascore+0.3*novelscore+0.5*dsscore
#     valid_score_list = score_list.tolist()
#     score_list = [-1]*len(neg_prop)
#     pos = 0
#     for idx,value in enumerate(neg_prop):
#         if value:
#             continue
#         else:
#             score_list[idx] = valid_score_list[pos]
#             pos+=1
#     return score_list
def evaluatemose(protein, smiles, mols=None,train_fps=None):
    df = pd.DataFrame()
    num_mols = len(smiles)
    if num_mols==0:
        return {'validity': 0, 'uniqueness': 0,
            'novelty': 0, 'top_ds': 0, 'hit': 0,"avgscore":0,"avgds":0,"avgqed":0,"avgsa":0}
    # remove empty molecules
    while True:
        if '' in smiles:
            idx = smiles.index('')
            del smiles[idx]
            if mols is not None:
                del mols[idx]
        else:
            break
    df['smiles'] = smiles
    validity = len(df) / num_mols

    if mols is None:
        df['mol'] = [Chem.MolFromSmiles(s) for s in smiles]
    else:
        df['mol'] = mols

    uniqueness = len(set(df['smiles'])) / len(df)
    get_novelty_in_df(df,1.,train_fps)
    novelty = len(df[df['sim'] < 0.6]) / len(df)

    df = df.drop_duplicates(subset=['smiles'])
    before_num = len(df)
    df[protein] = vinadock(protein,df["smiles"].tolist())
    df = df[~df[protein].isin([-1])]
    after_num = len(df)
    num_mols = num_mols-(before_num-after_num)

    df['qed'] = get_scores('qed', df['mol'])

    df['sa'] = get_scores('sa', df['mol'])
    avgscore = ((df[protein]/10)*df["qed"]*df["sa"]).mean()
    avgds = (df[protein]/10).mean()
    avgqed = df["qed"].mean()
    avgsa = df["sa"].mean()
    del df['mol']
    # df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf': hit_thr = 10.3
    else: raise ValueError('Wrong target protein')
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.6]
    df = df.sort_values(by=[protein], ascending=False)
    num_top5 = int(num_mols * 0.05)
    
    top_ds = df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std()
    hit = len(df[df[protein] > hit_thr]) / (num_mols+1e-6)
    
    return {'validity': validity, 'uniqueness': uniqueness,
            'novelty': novelty, 'top_ds': top_ds, 'hit': hit,"avgscore":avgscore,"avgds":avgds,"avgqed":avgqed,"avgsa":avgsa}

def evaluate_baseline(df, csv_dir, protein):
    from moses.utils import get_mol
    
    num_mols = 3000

    drop_idx = []
    mols = []
    for i, smiles in enumerate(df['smiles']):
        mol = get_mol(smiles)
        if mol is None:
            drop_idx.append(i)
        else:
            mols.append(mol)
    df = df.drop(drop_idx)
    df['mol'] = mols
    print(f'Validity: {len(df) / num_mols}')
    
    df['smiles'] = [Chem.MolToSmiles(m) for m in df['mol']]      # canonicalize

    print(f'Uniqueness: {len(set(df["smiles"])) / len(df)}')
    get_novelty_in_df(df)
    print(f"Novelty (sim. < 0.4): {len(df[df['sim'] < 0.4]) / len(df)}")

    df = df.drop_duplicates(subset=['smiles'])

    if not protein in df.keys():
        df[protein] = get_scores(protein, df['mol'])

    if not 'qed' in df.keys():
        df['qed'] = get_scores('qed', df['mol'])

    if not 'sa' in df.keys():
        df['sa'] = get_scores('sa', df['mol'])

    del df['mol']
    # df.to_csv(f'{csv_dir}.csv', index=False)

    if protein == 'parp1': hit_thr = 10.
    elif protein == 'fa7': hit_thr = 8.5
    elif protein == '5ht1b': hit_thr = 8.7845
    elif protein == 'jak2': hit_thr = 9.1
    elif protein == 'braf' : hit_thr = 10.3
    
    df = df[df['qed'] > 0.5]
    df = df[df['sa'] > (10 - 5) / 9]
    df = df[df['sim'] < 0.4]
    df = df.sort_values(by=[protein], ascending=False)

    num_top5 = int(num_mols * 0.05)

    top_ds = df.iloc[:num_top5][protein].mean(), df.iloc[:num_top5][protein].std()
    hit = len(df[df[protein] > hit_thr]) / num_mols
    
    print(f'Novel top 5% DS (QED > 0.5, SA < 5, sim. < 0.4): '
          f'{top_ds[0]:.4f} ± {top_ds[1]:.4f}')
    print(f'Novel hit ratio (QED > 0.5, SA < 5, sim. < 0.4): {hit * 100,:.4f} %')

if __name__=="__main__":
    smiles = ["Cc1noc(CCCN2CCC(=O)N(c3ccccc3O)CC2)n1","CC(=O)c1cc2ccc1c(=O)n2-c1ccc(C(=O)NC(C)C)s1"]
    print(gen_score_list('parp1',smiles))