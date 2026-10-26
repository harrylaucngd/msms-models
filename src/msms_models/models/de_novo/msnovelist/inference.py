"""MIST posteriors to the Morgan-4096 ctMSNovelist decoder."""

import re

import numpy as np

from ....encoders.mist import get_posteriors

ELEMENTS = ('C', 'F', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S', 'H')
MODEL_CONFIG = {'fp_dim': 4096, 'formula_dim': 10}


def formula_vector(formula):
    """A neutral formula's counts in the checkpoint's element order."""
    counts = {}
    for element, count in re.findall(r'([A-Z][a-z]?)([0-9]*)', formula):
        counts[element] = counts.get(element, 0) + int(count or 1)
    unsupported = set(counts) - set(ELEMENTS)
    if unsupported:
        raise ValueError(f'MSNovelist formula vocabulary does not contain {sorted(unsupported)}')
    return np.array([counts.get(e, 0) for e in ELEMENTS], dtype=np.float32)


def rank_candidates(smiles, scores, top_k=10):
    """Highest-score valid candidate per connectivity key; stable ties."""
    from ....evaluation.chemistry import canonical_smiles, inchikey14

    pairs = [(s, float(v)) for s, v in zip(smiles, scores, strict=True) if np.isfinite(v)]
    ranked = sorted(pairs, key=lambda x: -x[1])
    seen, candidates, retained_scores = set(), [], []
    for value, score in ranked:
        canonical = canonical_smiles(value)
        if not canonical:
            continue
        key = inchikey14(canonical)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(canonical)
        retained_scores.append(float(score))
        if len(candidates) == top_k:
            break
    return candidates, retained_scores


class MSNovelistDecoder:
    def __init__(self, checkpoint, *, device='cpu', beam_size=100):
        import torch

        from .vendor.model_msnovelist import DecoderModel, EncoderModel

        self.device = torch.device(device)
        self.beam_size = beam_size
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.encoder = EncoderModel(config=MODEL_CONFIG)
        self.decoder = DecoderModel(config=MODEL_CONFIG)
        for model in (self.encoder, self.decoder):
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()

    def generate(self, fingerprints, formulas):
        import torch

        from .vendor.model_msnovelist.beam_model.beam_search_decoder import BeamSearchDecoder

        count = len(fingerprints)
        beam = BeamSearchDecoder(
            self.encoder, self.decoder, steps=self.encoder.steps + 1, n=count,
            k=self.beam_size, kk=self.beam_size,
            config={'decoder_config': {'clip_invalid_counts': False}},
        )
        inputs = {
            'FP': torch.as_tensor(fingerprints, device=self.device),
            'MF': torch.as_tensor(formulas, device=self.device),
        }
        with torch.inference_mode():
            state = self.encoder({k: v.repeat_interleave(self.beam_size, 0)
                                  for k, v in inputs.items()})
            sequences, tokens, scores = beam.decode_beam(state)
            traces, scores, _ = beam.beam_traceback(sequences, tokens, scores)
            smiles = beam.sequence_ytoc(traces)
        return [(smiles[i * self.beam_size:(i + 1) * self.beam_size], scores[i].tolist())
                for i in range(count)]


def predict(config, queries):
    import torch

    beam_size = int(config.parameters.get('beam_size', 100))
    top_k = int(config.parameters.get('top_k', 10))
    batch_size = int(config.parameters.get('batch_size', 32))
    posterior = np.asarray(get_posteriors(config, queries), dtype=np.float32)
    torch.set_num_threads(config.workers)
    decoder = MSNovelistDecoder(config.require_assets('decoder_checkpoint')['decoder_checkpoint'],
                                device=config.device, beam_size=beam_size)
    rows, supported = {}, []
    for index, query in enumerate(queries):
        try:
            supported.append((index, formula_vector(query['formula'])))
        except ValueError as error:
            rows[query['id']] = {'id': query['id'], 'candidates': [], 'status': 'unsupported_formula',
                                 'reason': str(error)}
    for start in range(0, len(supported), batch_size):
        batch = supported[start:start + batch_size]
        generated = decoder.generate(posterior[[i for i, _ in batch]],
                                     np.stack([formula for _, formula in batch]))
        for (index, _), (smiles, scores) in zip(batch, generated, strict=True):
            candidates, scores = rank_candidates(smiles, scores, top_k)
            rows[queries[index]['id']] = {'id': queries[index]['id'], 'candidates': candidates,
                                          'scores': scores, 'status': 'ok'}
    return [rows[q['id']] for q in queries]
