"""Download the official MassSpecGym files at a fixed revision."""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen


def download_massspecgym(output, *, candidates=None, candidate_version="original"):
    source = json.loads(files(__package__).joinpath("massspecgym.json").read_text())
    keys = ["spectra"]
    if candidates:
        pools = ("formula", "mass") if candidates == "both" else (candidates,)
        keys += [f"candidates-{candidate_version}-{pool}" for pool in pools]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key in keys:
        path = source["files"][key]
        destination = output / Path(path).name
        if not destination.exists():
            url = f"{source['repository']}/resolve/{source['revision']}/{path}"
            with urlopen(url) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        downloaded.append(str(destination))
    return {"revision": source["revision"], "files": downloaded}
