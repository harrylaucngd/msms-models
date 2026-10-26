"""Read the prepared legacy Morgan-4096 SQLite database without modifying it."""

import pickle
import sqlite3
from pathlib import Path

import numpy as np

from .inference import ELEMENTS


def decode_formula(blob):
    """The stored formula, a pickled ``collections.Counter``, in the checkpoint's element order."""
    counts = pickle.loads(blob)
    return np.array([counts.get(e, 0) for e in ELEMENTS], dtype=np.float32)


class FoldDataset:
    def __init__(self, database, fold=0, *, validation=False):
        self.uri = Path(database).expanduser().resolve().as_uri() + '?mode=ro'
        with self.connect() as conn:
            groups = [f'fold{i}' for i in range(10) if (i == fold) == validation]
            marks = ','.join('?' for _ in groups)
            self.ids = [r[0] for r in conn.execute(
                f'SELECT id FROM compounds WHERE grp IN ({marks}) ORDER BY perm_order, id', groups
            )]

    def connect(self):
        # A closing context also releases the connection on a failed query.
        from contextlib import closing

        return closing(sqlite3.connect(self.uri, uri=True))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from .vendor import tokens_process as tokens

        with self.connect() as conn:
            row = conn.execute('SELECT fp, mf, smiles FROM compounds WHERE id=?',
                               (self.ids[index],)).fetchone()
        fp_blob, formula_blob, smiles = row
        fp = np.frombuffer(fp_blob, dtype=np.uint8)  # 4096 unpacked bits
        x, y = tokens.smiles2_gt_lab(smiles)
        return fp.astype(np.float32), decode_formula(formula_blob), x, y.astype(np.int64)
