"""Build the SQLite training database of the Morgan-4096 MSNovelist decoder.

Each molecule of a SMILES list becomes one ``compounds`` row: its non-isomeric RDKit SMILES,
its unpacked Morgan radius-2/4096 fingerprint, its formula counts and one of ten folds.
Molecules the decoder cannot write (tokens outside its vocabulary, elements outside
``C F I Cl N O P Br S H``, more than 126 tokens) are skipped.

    python -m msms_models.models.de_novo.msnovelist.build_database \\
        --smiles fp2mol_train.txt --output data/msnovelist/train.db
"""

import argparse
import collections
import json
import logging
import pickle
import sqlite3
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .inference import ELEMENTS


def row(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    from .vendor.tokens_process import definitions, functions

    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    tokens = functions.spilt_smiles(functions.replace_smiles_ele(smiles))
    if len(tokens) > definitions.SEQUENCE_LEN - 2 or any(t not in definitions.VOC2IDX for t in tokens):
        return None
    counts = collections.Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
    if any(e not in ELEMENTS for e in counts):
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    fp = generator.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return smiles, fp.tobytes(), pickle.dumps(counts)


def build(smiles_path, output, *, seed=42, workers=16):
    smiles = [s.split()[0] for s in Path(smiles_path).read_text().splitlines() if s.strip()]
    with Pool(workers) as pool:
        rows = [r for r in pool.imap(row, smiles, chunksize=2_000) if r is not None]
    # Deduplicate by SMILES, then assign folds and a training order at random.
    rows = list({r[0]: r for r in rows}.values())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as conn:
        conn.execute('CREATE TABLE compounds (id INTEGER PRIMARY KEY, smiles TEXT, fp BLOB, '
                     'mf BLOB, grp TEXT, perm_order INTEGER)')
        conn.executemany(
            'INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?)',
            ((i, s, fp, mf, f'fold{order[i] % 10}', int(order[i]))
             for i, (s, fp, mf) in enumerate(rows)),
        )
    return {'molecules': len(smiles), 'kept': len(rows), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--smiles', required=True, help='One SMILES per line')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    print(json.dumps(build(args.smiles, args.output, seed=args.seed, workers=args.workers)))


if __name__ == '__main__':
    main()
