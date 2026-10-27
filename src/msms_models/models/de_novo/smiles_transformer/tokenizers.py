"""The SMILES (byte-level BPE) and SELFIES (word-level) tokenizers of MassSpecGym (``models/tokenizers.py``).

Built as upstream builds them, then saved with the checkpoint as Hugging Face ``tokenizers``
JSON, which keeps the vocabulary, the special tokens, padding, truncation and the
``<s> … </s>`` template.
"""

from __future__ import annotations

PAD, SOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


def _add_special_tokens(tokenizer, max_len):
    """``SpecialTokensBaseTokenizer``: special tokens, right padding and truncation to
    ``max_len``, and ``<s> … </s>`` around every sequence."""
    from tokenizers import processors

    tokenizer.add_special_tokens([PAD, SOS, EOS, UNK])
    tokenizer.enable_padding(direction="right", pad_token=PAD, pad_id=tokenizer.token_to_id(PAD),
                             length=max_len)
    tokenizer.enable_truncation(max_len)
    sos, eos = tokenizer.token_to_id(SOS), tokenizer.token_to_id(EOS)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{SOS} $A {EOS}", pair=f"{SOS} $A {EOS} {SOS} $B {EOS}",
        special_tokens=[(SOS, sos), (EOS, eos)],
    )
    return tokenizer


class Tokenizer:
    """Encodes SMILES to token ids and decodes them back, for ``kind`` ``smiles-bpe`` or ``selfies``."""

    def __init__(self, tokenizer, kind):
        self.tokenizer, self.kind = tokenizer, kind

    @classmethod
    def train(cls, kind, train_smiles, molecules=(), max_len=None):
        """``SmilesBPETokenizer`` (``molecules`` then ``train_smiles``) or ``SelfiesTokenizer``."""
        if kind == "smiles-bpe":
            from tokenizers import ByteLevelBPETokenizer

            bpe = ByteLevelBPETokenizer()
            bpe.train_from_iterator(list(molecules) + list(train_smiles))
            tokenizer = bpe._tokenizer
        else:
            import selfies as sf
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import models

            # The alphabet is a set, so repeated training molecules change nothing.
            selfies = [sf.encoder(s, strict=False) for s in set(train_smiles)]
            alphabet = sorted(sf.get_alphabet_from_selfies(selfies))
            vocab = {symbol: i for i, symbol in enumerate(alphabet)}
            vocab[UNK] = len(vocab)
            tokenizer = HFTokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
        return cls(_add_special_tokens(tokenizer, max_len), kind)

    @classmethod
    def load(cls, path, kind):
        from tokenizers import Tokenizer as HFTokenizer

        return cls(HFTokenizer.from_file(str(path)), kind)

    def save(self, path):
        self.tokenizer.save(str(path))

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode_batch(self, smiles):
        if self.kind == "smiles-bpe":
            return [e.ids for e in self.tokenizer.encode_batch(list(smiles))]
        import selfies as sf

        tokens = [list(sf.split_selfies(sf.encoder(s, strict=False))) for s in smiles]
        return [e.ids for e in self.tokenizer.encode_batch(tokens, is_pretokenized=True)]

    def decode_batch(self, ids):
        """Special tokens are dropped, but tokens after ``</s>`` are kept, as upstream."""
        strings = self.tokenizer.decode_batch(ids, skip_special_tokens=True)
        if self.kind == "smiles-bpe":
            return strings
        import selfies as sf

        return [sf.decoder(s.replace(" ", "")) for s in strings]
