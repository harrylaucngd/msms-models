"""Read the formula-grouped spectrum parquet exports released with MS-GPT."""

from pathlib import Path


def iter_annotated_spectra(path, columns=None):
    import pyarrow.parquet as pq

    path = Path(path)
    for file in [path] if path.is_file() else sorted(path.rglob("*.parquet")):
        for row in pq.read_table(file, columns=columns).to_pylist():
            if not isinstance(row["identifier"], list):
                yield row
                continue
            for i in range(len(row["identifier"])):
                yield {key: value[i] if isinstance(value, list) else value for key, value in row.items()}
