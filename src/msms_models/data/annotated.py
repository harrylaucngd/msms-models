"""Read the formula-grouped spectrum exports released with MS-GPT."""

from pathlib import Path


def iter_annotated_spectra(path, columns=None):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Parquet spectrum inputs require pyarrow (included in .[ms-gpt])") from error

    path = Path(path)
    paths = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No annotated spectrum parquet files in {path}")
    for file in paths:
        for row in pq.read_table(file, columns=columns).to_pylist():
            identifiers = row["identifier"]
            if not isinstance(identifiers, list):
                yield row
                continue
            for i in range(len(identifiers)):
                yield {
                    key: value[i] if isinstance(value, list) else value
                    for key, value in row.items()
                }
