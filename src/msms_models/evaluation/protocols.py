"""The fixed evaluation protocols behind the paper's Tables 3 and 5.

Both follow MassSpecGym: InChIKey-14 identity, Morgan radius-2/2048 Tanimoto
without chirality, and myopic MCES with threshold 15 and the stronger bound,
where an invalid or missing prediction has distance 100. Means are over every
test spectrum; 99.9% BCa intervals use 20,000 spectrum-level resamples.
"""

from dataclasses import replace

from .metrics import MetricProtocol

PROTOCOLS = {
    # MassSpecGym retrieval with bonus (formula-known candidates).
    "table3": MetricProtocol(task="retrieval", cutoffs=(1, 5, 20), mces=True),
    # MassSpecGym de novo generation with bonus; raw top-10 slots, no deduplication.
    "table5": MetricProtocol(task="de-novo", cutoffs=(1, 10), mces=True),
}


def get_protocol(name: str, **overrides) -> MetricProtocol:
    return replace(PROTOCOLS[name], **overrides)
