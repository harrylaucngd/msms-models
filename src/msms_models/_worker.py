"""Backend subprocess bootstrap, without importing another environment's site-packages."""

import importlib.util
import sys
from pathlib import Path


def main():
    if __package__ in (None, ""):
        package = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "msms_models", package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["msms_models"] = module
        spec.loader.exec_module(module)
    from msms_models.api import load_request, predict
    from msms_models.io import atomic_json, load_queries, write_jsonl
    from msms_models.provenance import environment_record

    request = load_request(sys.argv[1])
    atomic_json(request.output / "environment.json", environment_record())
    queries = load_queries(request.queries)
    write_jsonl(request.output / "predictions.jsonl", predict(request, queries))


if __name__ == "__main__":
    main()
