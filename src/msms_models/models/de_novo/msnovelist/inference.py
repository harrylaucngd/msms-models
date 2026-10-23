"""MIST posteriors to the Morgan-4096 ctMSNovelist decoder."""

import re

import numpy as np

from ....encoders.mist import get_posteriors

ELEMENTS = ('C', 'F', 'I', 'Cl', 'N', 'O', 'P', 'Br', 'S', 'H')
MODEL_CONFIG = {'fp_dim': 4096, 'formula_dim': 10}


def formula_vector(formula):
    """Encode a neutral formula in the checkpoint's element order."""
    parts = re.findall(r'([A-Z][a-z]?)([0-9]*)', formula)
    if not parts or ''.join(e + n for e, n in parts) != formula:
        raise ValueError(f'Invalid neutral molecular formula: {formula!r}')
    counts = {}
    for element, count in parts:
        if element not in ELEMENTS:
            raise ValueError(f'Element {element} is outside the MSNovelist vocabulary')
        value = int(count or 1)
        if value <= 0:
            raise ValueError(f'Nonpositive atom count in {formula!r}')
        counts[element] = counts.get(element, 0) + value
    vector = np.array([counts.get(e, 0) for e in ELEMENTS], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError('Formula counts exceed the supported numeric range')
    return vector


def rank_candidates(smiles, scores, top_k=10):
    """Highest-score valid candidate per connectivity key; stable ties."""
    from ....evaluation.chemistry import canonical_smiles, inchikey14

    if top_k < 1:
        raise ValueError('top_k must be positive')
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
    if min(beam_size, top_k, batch_size, config.workers) < 1 or top_k > beam_size:
        raise ValueError('Positive sizes/workers and top_k <= beam_size are required')
    if not queries:
        return []
    formulas = np.stack([formula_vector(q['formula']) for q in queries])
    posterior = np.asarray(get_posteriors(config, queries), dtype=np.float32)
    if (posterior.shape != (len(queries), 4096) or not np.isfinite(posterior).all()
            or np.any((posterior < 0) | (posterior > 1))):
        raise ValueError('Expected finite MIST probabilities with shape (queries, 4096) in [0, 1]')
    torch.set_num_threads(config.workers)
    torch.manual_seed(config.seed)
    decoder = MSNovelistDecoder(config.require_assets('decoder_checkpoint')['decoder_checkpoint'],
                               device=config.device, beam_size=beam_size)
    rows = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        generated = decoder.generate(posterior[start:start + batch_size],
                                     formulas[start:start + batch_size])
        for query, (smiles, scores) in zip(batch, generated, strict=True):
            candidates, scores = rank_candidates(smiles, scores, top_k)
            rows.append({'id': query['id'], 'candidates': candidates, 'scores': scores,
                         'status': 'ok'})
    return rows
