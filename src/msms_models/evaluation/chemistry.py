"""Declared structure identities; no implicit tautomer or salt transformations."""

from functools import lru_cache

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


@lru_cache(maxsize=100_000)
def canonical_smiles(smiles: str | None, tautomer: bool = False) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    if tautomer:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=100_000)
def inchikey14(smiles: str | None, tautomer: bool = False) -> str | None:
    value = canonical_smiles(smiles, tautomer)
    if value is None:
        return None
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(value))
    return key[:14] if key else None


def valid_smiles(smiles, *, tautomer: bool = False) -> bool:
    return canonical_smiles(smiles, tautomer) is not None


@lru_cache(maxsize=50_000)
def fingerprint(smiles: str, bits: int = 2048, tautomer: bool = False):
    # The benchmark fingerprints the input molecular graph directly. A SMILES
    # round-trip after removing stereochemistry can drop explicit hydrogens
    # that were retained for double-bond stereochemistry, changing Morgan bits.
    value = canonical_smiles(smiles, True) if tautomer else smiles
    molecule = Chem.MolFromSmiles(value) if value else None
    if molecule is None:
        raise ValueError(f"Invalid SMILES {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=bits, includeChirality=False
    )
    return generator.GetFingerprint(molecule)


def tanimoto2048(target: str, predicted: str | None, *, tautomer: bool = False) -> float:
    if not valid_smiles(predicted, tautomer=tautomer):
        return 0.0
    return float(
        DataStructs.TanimotoSimilarity(
            fingerprint(target, 2048, tautomer), fingerprint(predicted, 2048, tautomer)
        )
    )
