# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py
from __future__ import print_function
from rdkit import Chem
import rdkit.Chem.QED as QED
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



import math
import os.path as op
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.six import iteritems
from scorer.vinadock import vinadock
_fscores = None
#!/usr/bin/env python
import os

def standardize_mols(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        print('standardize_mols error')
        return None
def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro
def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2  # <- 2 is the *radius* of the circular fingerprint
    )
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (0. - sizePenalty - stereoPenalty -
              spiroPenalty - bridgePenalty - macrocyclePenalty)

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def get_scores(objective, mols, standardize=True):
    if standardize:
        mols = [standardize_mols(mol) for mol in mols]
    mols_valid = [mol for mol in mols if mol is not None]
    
    scores = [get_score(objective, mol) for mol in mols_valid]
    scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
    
    return scores


def get_score(objective, mol):
    try:
        if objective == 'qed':
            return QED.qed(mol)
        elif objective == 'sa':
            x = calculateScore(mol)
            return (10. - x) / 9.   # normalized to [0, 1]
        else:
            raise NotImplementedError
    except (ValueError, ZeroDivisionError):
        return 0.

if __name__=="__main__":
    smile = "C1CC1NCCc2nnc1s2COc1ccc(N(C)C(C)=O)cc1CC1(C)CC2(C#N)CC3CCC(C3)C1(C)C2CC(C)N1CC2CCC(CC1C(=O)Nc1nc(-c3ccco3)n(C)n1)O2CCCN(CCCN1C(=O)NCCC1CC)c1nccc(NC2CC2)n1c1ccc"
    mol = Chem.MolFromSmiles(smile)
    