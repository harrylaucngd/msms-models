import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
from msms_models._vendor.iceberg import common

def fp_from_mol(mol):
    """fp_from_mol."""
    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint

def tautomerize_smi(smi):
    """tautomerize_smi, as roundtripped through InChi"""
    mol = Chem.MolFromSmiles(smi)
    inchi = Chem.MolToInchi(mol)
    mol = common.chem_utils.canonical_mol_from_inchi(inchi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
