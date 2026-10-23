"""Read the prepared legacy Morgan-4096 SQLite database without modifying it."""

import io
import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from .inference import ELEMENTS


class _FormulaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'Counter'):
            return Counter
        raise pickle.UnpicklingError(f'Unsupported formula pickle global: {module}.{name}')


def decode_formula(blob):
    counts = _FormulaUnpickler(io.BytesIO(blob)).load()
    if not isinstance(counts, dict) or not counts:
        raise ValueError('Expected a nonempty molecular-formula counter')
    for element, count in counts.items():
        if element not in ELEMENTS or type(count) is not int or count < 0:
            raise ValueError(f'Unsupported formula entry: {element!r}: {count!r}')
    vector = np.array([counts.get(e, 0) for e in ELEMENTS], dtype=np.float32)
    if not np.isfinite(vector).all() or not vector.any():
        raise ValueError('Invalid formula counts')
    return vector


class FoldDataset:
    def __init__(self, database, fold=0, *, validation=False):
        if not 0 <= fold <= 9:
            raise ValueError('fold must be between 0 and 9')
        self.uri = Path(database).expanduser().resolve().as_uri() + '?mode=ro'
        with self.connect() as conn:
            columns = {r[1] for r in conn.execute('PRAGMA table_info(compounds)')}
            if not {'id', 'fp', 'smiles', 'mf', 'grp', 'perm_order'} <= columns:
                raise ValueError('Missing required columns in compounds table')
            groups = [f'fold{i}' for i in range(10) if (i == fold) == validation]
            marks = ','.join('?' for _ in groups)
            self.ids = [r[0] for r in conn.execute(
                f'SELECT id FROM compounds WHERE grp IN ({marks}) ORDER BY perm_order, id', groups
            )]
        if not self.ids:
            raise ValueError('Selected training/validation split is empty')

    def connect(self):
        # A closing context also releases the connection on a failed query.
        from contextlib import closing

        return closing(sqlite3.connect(self.uri, uri=True))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from rdkit import Chem

        from .vendor import tokens_process as tokens

        with self.connect() as conn:
            row = conn.execute('SELECT fp, mf, smiles FROM compounds WHERE id=?',
                               (self.ids[index],)).fetchone()
        if row is None:
            raise ValueError(f'Missing compound id {self.ids[index]}')
        fp_blob, formula_blob, smiles = row
        fp = np.frombuffer(fp_blob, dtype=np.uint8)
        if fp.shape != (4096,) or np.any(fp > 1):
            raise ValueError('Expected 4096 unpacked binary fingerprint bytes')
        if not isinstance(smiles, str) or not smiles or Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f'Invalid training SMILES for id {self.ids[index]}')
        parts = tokens.spilt_smiles(tokens.replace_smiles_ele(smiles))
        if len(parts) > 126 or any(t not in tokens.VOC2IDX for t in parts):
            raise ValueError(f'SMILES exceeds token vocabulary/length for id {self.ids[index]}')
        x, y = tokens.smiles2_gt_lab(smiles)
        return fp.astype(np.float32), decode_formula(formula_blob), x, y.astype(np.int64)
