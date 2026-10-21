"""ChemBERTa with a two-class head, and the SMILES strings it reads.

Paper Appendix A.4, method 3. A pretrained SMILES transformer learns to tell
query molecules from their retrieval decoys using the SMILES string alone.
"""

from __future__ import annotations

from functools import lru_cache

BASE_MODEL = "DeepChem/ChemBERTa-77M-MLM"


@lru_cache(maxsize=1_000_000)
def rdkit_smiles(smiles: str) -> str:
    """RDKit canonical SMILES, as used for the corrected (canonicalized) candidate sets."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def text(smiles: str, canonical: bool) -> str:
    return rdkit_smiles(smiles) if canonical else smiles


def load(path_or_name: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_name, num_labels=2)
    return tokenizer, model.to(device)


def positive_probability(tokenizer, model, smiles: list[str], *, max_length: int, batch_size: int):
    import torch

    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            batch = tokenizer(
                smiles[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(model.device)
            values += torch.softmax(model(**batch).logits.float(), -1)[:, 1].tolist()
    return values
