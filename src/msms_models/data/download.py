"""Download immutable public assets and verify their published content hashes."""

from __future__ import annotations

import hashlib
import json
import tempfile
from importlib.resources import files
from pathlib import Path
from urllib.request import Request, urlopen

from .io import atomic_json


def catalog():
    return json.loads(files(__package__).joinpath("massspecgym.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, destination, *, expected_sha256, expected_size=None):
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for existing {destination}; move it before retrying"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest, size = hashlib.sha256(), 0
            request = Request(url, headers={"User-Agent": "msms-models"})
            with urlopen(request, timeout=60) as response:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
        if expected_size is not None and size != expected_size:
            raise ValueError(f"Download size mismatch for {destination}: {size} != {expected_size}")
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Download checksum mismatch for {destination}")
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_massspecgym(output, *, candidates=None, candidate_version="original"):
    source = catalog()
    keys = ["spectra"]
    if candidates:
        pools = ("formula", "mass") if candidates == "both" else (candidates,)
        keys += [f"candidates-{candidate_version}-{pool}" for pool in pools]
    output = Path(output)
    entries = []
    for key in keys:
        item = source["files"][key]
        url = f"{source['repository']}/resolve/{source['revision']}/{item['path']}"
        destination = output / Path(item["path"]).name
        fetch(url, destination, expected_sha256=item["sha256"], expected_size=item["size"])
        entries.append({"id": key, "local_path": str(destination.resolve()), "url": url, **item})
    manifest = {
        "dataset": "msg",
        "revision": source["revision"],
        "candidate_version": candidate_version if candidates else None,
        "files": entries,
    }
    atomic_json(output / "download_manifest.json", manifest)
    return manifest
