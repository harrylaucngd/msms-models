from __future__ import annotations
"""
ICEBERG-guided inference-time scaling sampler for DLM.

This module provides the IcebergSampler class that extends the standard FRIGID-base
sampler with ICEBERG-guided refinement for improved structure elucidation.

The algorithm ("Spectrum-Error Guided Refinement"):
1. Generate diverse candidate molecules (B samples per round)
2. Select top K unique candidates by fingerprint similarity
3. Simulate mass spectra with ICEBERG
4. Identify hallucinated peaks and map to atoms/tokens
5. Create M masked versions per molecule and refine with DLM
6. Repeat for R rounds
7. Rank all unique candidates by fingerprint similarity
"""

import os
import sys
import re
import uuid
import shutil
import tempfile
import hashlib
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any, Callable, Union
from collections import defaultdict

import time

import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors



import msms_models.methods.iceberg.vendor.common as common
from msms_models.methods.iceberg.vendor.common.chem_utils import VALID_ELEMENTS

from msms_models.methods.frigid.vendor.sampler import Sampler
from msms_models.methods.frigid.vendor.utils.masking_utils import (
    BaseMaskingStrategy,
    SimpleMaskingStrategy,
    create_masking_strategy,
)
from msms_models.methods.frigid.vendor.utils.benchmark_utils import (
    normalize_formula,
    get_molecular_formula,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
    predict_token_count,
)


@dataclass
class IcebergConfig:
    """Configuration for ICEBERG spectrum prediction."""
    gen_ckpt: str = ''
    inten_ckpt: str = ''
    python_path: str = 'python'
    cuda_devices: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 8
    num_gpu_workers: int = 1
    num_cpu_workers: int = 8
    ppm: int = 20
    num_bins: int = 15000
    upper_limit: int = 1500
    adduct: str = '[M+H]+'
    sparse_k: int = 100
    max_nodes: int = 100
    threshold: float = 0.0


@dataclass
class ScalingConfig:
    """Configuration for inference-time scaling."""
    batch_size: int = 128  # B: total samples per round
    num_unique_to_refine: int = 8  # K: unique molecules to refine
    masks_per_molecule: int = 12  # M: masks per molecule
    num_rounds: int = 50  # R: number of refinement rounds
    mask_prob: float = 0.75  # Probability of masking each bad token
    top_k_halluc_peaks: int = 10  # Number of hallucinated peaks to consider
    halluc_inten_threshold: float = 0.05  # Intensity threshold for hallucination
    softmax_temp: float = 1.0  # Temperature for sampling
    randomness: float = 10.0  # Randomness factor
    collision_energies: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90])
    nce: bool = True  # Whether collision energies are normalized
    seed: int = 42  # Base seed; expanded deterministically by spectrum and round
    # A base-only benchmark can publish the frozen Round-1 Top-K without paying
    # for ICEBERG until a later refinement campaign is requested.  A resumed
    # run detects this marker and prepares the missing Top-K spectra before it
    # executes the next DLM correction round.
    defer_next_round_iceberg: bool = False


def remove_stereo_from_smiles(smiles: str) -> Optional[str]:
    """Remove stereochemistry from a SMILES string using ms_pred utility."""
    try:
        return common.rm_stereo(smiles)
    except Exception:
        return None


def get_inchikey_no_stereo(smiles: str) -> Optional[str]:
    """Get InChI key first block (no stereochemistry) for a SMILES."""
    try:
        # Remove stereochemistry using ms_pred utility
        smi_no_stereo = common.rm_stereo(smiles)
        if smi_no_stereo is None:
            return None
        # Get full InChI key
        inchi_key = common.inchikey_from_smiles(smi_no_stereo)
        # Return first block (connectivity layer)
        return get_inchikey_first_block(inchi_key) if inchi_key else None
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES corresponds to a valid molecule."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        Chem.SanitizeMol(mol)
        
        # Check for fragmented molecules
        if len(Chem.GetMolFrags(mol)) > 1:
            return False
        
        # Check for chemistry problems
        if Chem.DetectChemistryProblems(mol):
            return False
        
        # Check for unsupported elements
        if not all(atom.GetSymbol() in VALID_ELEMENTS for atom in mol.GetAtoms()):
            return False
        
        # Check molecular weight
        if Descriptors.ExactMolWt(mol) > 1500:
            return False
        
        return True
    except Exception:
        return False


def normalize_instrument_type(instrument: Optional[str]) -> str:
    """
    Normalize instrument type to one of: 'Orbitrap', 'QTOF', or 'Unknown'.
    
    Handles cases where instrument type is not exactly 'QTOF' but contains 'qtof',
    or contains 'orbitrap', etc.
    
    Args:
        instrument: Raw instrument type string (e.g., 'QTOF (LCMS)', 'Orbitrap', 'Q-TOF', etc.)
        
    Returns:
        Normalized instrument type: 'Orbitrap', 'QTOF', or 'Unknown'
    """
    if instrument is None or (isinstance(instrument, float) and np.isnan(instrument)):
        return "Orbitrap" # for some reason some MSG data has NaN instruments, but MSG ICEBERG model doesn't support unknown
    
    if not isinstance(instrument, str):
        return 'Unknown'
    
    instrument_lower = instrument.lower().strip()
    
    # Check for empty string
    if not instrument_lower:
        return 'Unknown'
    
    # Check for QTOF variants (Q-TOF, qTOF, QTOF (LCMS), etc.)
    if 'qtof' in instrument_lower or 'q-tof' in instrument_lower:
        return 'QTOF'
    
    # Check for Orbitrap variants
    if 'orbitrap' in instrument_lower:
        return 'Orbitrap'
    
    # Unknown otherwise
    return 'Unknown'


class IcebergPredictionCache:
    """
    Cache for ICEBERG predictions to avoid redundant computation.

    Keys are (inchi_key_no_stereo, collision_energy, instrument, ionization) tuples.
    Values are CompositeMassSpec objects.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, int, str, str], common.CompositeMassSpec] = {}
        self._smiles_to_ikey: Dict[str, str] = {}  # Cache SMILES -> InChI key mapping
    
    def get_ikey(self, smiles: str) -> Optional[str]:
        """Get cached InChI key or compute it."""
        if smiles not in self._smiles_to_ikey:
            ikey = get_inchikey_no_stereo(smiles)
            if ikey:
                self._smiles_to_ikey[smiles] = ikey
        return self._smiles_to_ikey.get(smiles)
    
    def get(self, smiles: str, collision_energy: int, instrument: str, ionization: str = '[M+H]+') -> Optional[common.CompositeMassSpec]:
        """Get cached prediction if available."""
        ikey = self.get_ikey(smiles)
        if ikey is None:
            return None
        return self._cache.get((ikey, collision_energy, instrument, ionization))

    def has(self, smiles: str, collision_energy: int, instrument: str, ionization: str = '[M+H]+') -> bool:
        """Check if prediction is cached."""
        ikey = self.get_ikey(smiles)
        if ikey is None:
            return False
        return (ikey, collision_energy, instrument, ionization) in self._cache

    def put(self, smiles: str, collision_energy: int, instrument: str, pred_spec: common.CompositeMassSpec, ionization: str = '[M+H]+'):
        """Cache a prediction."""
        ikey = self.get_ikey(smiles)
        if ikey:
            self._cache[(ikey, collision_energy, instrument, ionization)] = pred_spec
    
    def get_uncached_pairs(
        self, 
        smiles_list: List[str], 
        collision_energies: List[int],
        instrument: str,
    ) -> Dict[int, List[str]]:
        """
        Get (collision_energy -> list of smiles) for uncached predictions.
        
        Returns a dict mapping collision energy to list of SMILES that need prediction
        for the given instrument type.
        """
        ce_to_smiles: Dict[int, List[str]] = defaultdict(list)
        seen_ikeys: Dict[int, Set[str]] = defaultdict(set)  # Avoid duplicate SMILES per CE
        
        for smiles in smiles_list:
            ikey = self.get_ikey(smiles)
            if ikey is None:
                continue
            
            for ce in collision_energies:
                if not self.has(smiles, ce, instrument) and ikey not in seen_ikeys[ce]:
                    ce_to_smiles[ce].append(smiles)
                    seen_ikeys[ce].add(ikey)
        
        return dict(ce_to_smiles)
    
    def get_uncached_pairs_grouped(
        self,
        smiles_instrument_ionization_triples: List[Tuple[str, str, str]],
        collision_energies: List[int],
    ) -> Dict[Tuple[int, str, str], List[str]]:
        """
        Get ((collision_energy, instrument, ionization) -> list of smiles) for uncached predictions.

        Args:
            smiles_instrument_ionization_triples: List of (SMILES, instrument, ionization) tuples
            collision_energies: List of collision energies to check

        Returns:
            Dict mapping (CE, instrument, ionization) tuple to list of SMILES needing prediction
        """
        ce_inst_ion_to_smiles: Dict[Tuple[int, str, str], List[str]] = defaultdict(list)
        seen_ikeys: Dict[Tuple[int, str, str], Set[str]] = defaultdict(set)

        for smiles, instrument, ionization in smiles_instrument_ionization_triples:
            ikey = self.get_ikey(smiles)
            if ikey is None:
                continue

            for ce in collision_energies:
                key = (ce, instrument, ionization)
                if not self.has(smiles, ce, instrument, ionization) and ikey not in seen_ikeys[key]:
                    ce_inst_ion_to_smiles[key].append(smiles)
                    seen_ikeys[key].add(ikey)

        return dict(ce_inst_ion_to_smiles)
    
    def __len__(self) -> int:
        return len(self._cache)


class SampleState:
    """
    Tracks the state of a single test sample across scaling rounds.

    Attributes:
        name: Spectrum name/identifier
        real_specs: Real composite mass spectrum
        target_fp: Target fingerprint (from MIST encoder)
        target_formula: Target molecular formula (or default formula when using candidates)
        instrument: Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
        ionization: Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+')
        candidate_formulas: Optional list of (formula, probability) tuples for sampling
        all_unique_smiles: Set of all unique SMILES generated (by InChI key)
        all_unique_ikeys: Set of InChI keys for uniqueness tracking
        smiles_to_similarity: Dict mapping SMILES -> Tanimoto similarity
        smiles_to_formula: Dict mapping SMILES -> normalized molecular formula
    """

    def __init__(
        self,
        name: str,
        real_specs: common.CompositeMassSpec,
        target_fp: np.ndarray,
        target_formula: str,
        instrument: str = 'Unknown',
        ionization: str = '[M+H]+',
        candidate_formulas: Optional[List[Tuple[str, float]]] = None,
    ):
        self.name = name
        self.real_specs = real_specs
        self.target_fp = target_fp
        self.target_formula = target_formula
        self.instrument = normalize_instrument_type(instrument)
        self.ionization = ionization
        self.candidate_formulas = candidate_formulas
        
        # Tracking across rounds
        self.all_unique_smiles: Set[str] = set()
        self.all_unique_ikeys: Set[str] = set()
        self.smiles_to_similarity: Dict[str, float] = {}
        self.smiles_to_formula: Dict[str, Optional[str]] = {}
    
    def sample_formula(self) -> str:
        """
        Sample a formula from candidate formulas if available, otherwise return target_formula.
        
        When candidate_formulas is provided, samples according to their probabilities.
        Otherwise, returns the fixed target_formula.
        
        Returns:
            A molecular formula string
        """
        if self.candidate_formulas is None or len(self.candidate_formulas) == 0:
            return self.target_formula
        
        formulas = [f for f, _ in self.candidate_formulas]
        probs = [p for _, p in self.candidate_formulas]
        
        # Normalize probabilities
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # Fallback to uniform
            probs = [1.0 / len(formulas)] * len(formulas)
        
        return np.random.choice(formulas, p=probs)
    
    def sample_formulas(self, n: int) -> List[str]:
        """
        Sample n formulas from candidate formulas.
        
        Args:
            n: Number of formulas to sample
            
        Returns:
            List of molecular formula strings
        """
        return [self.sample_formula() for _ in range(n)]
    
    def add_smiles(self, smiles: str, fp_bits: int = 4096, fp_radius: int = 2) -> bool:
        """
        Add a SMILES to the collection if it's unique.
        
        Returns True if the SMILES was added (is unique), False otherwise.
        """
        if not is_valid_smiles(smiles):
            return False
        
        ikey = get_inchikey_no_stereo(smiles)
        if ikey is None or ikey in self.all_unique_ikeys:
            return False
        
        self.all_unique_ikeys.add(ikey)
        self.all_unique_smiles.add(smiles)
        
        # Compute and cache similarity
        fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
        if fp is not None:
            sim = compute_tanimoto_similarity(self.target_fp, fp)
            self.smiles_to_similarity[smiles] = sim
        else:
            self.smiles_to_similarity[smiles] = 0.0
        self.smiles_to_formula[smiles] = get_molecular_formula(smiles)
        
        return True
    
    def add_smiles_batch(
        self, 
        smiles_list: List[str], 
        fp_bits: int = 4096, 
        fp_radius: int = 2
    ) -> int:
        """Add multiple SMILES, returns count of newly added unique SMILES."""
        count = 0
        for smiles in smiles_list:
            if self.add_smiles(smiles, fp_bits, fp_radius):
                count += 1
        return count
    
    def _ranking_formula_set(self) -> Set[str]:
        """Return the normalized formulae that receive ranking priority."""
        if self.candidate_formulas:
            values = (formula for formula, _probability in self.candidate_formulas)
        else:
            values = (self.target_formula,)
        return {
            normalized
            for formula in values
            if (normalized := normalize_formula(formula)) is not None
        }

    def _ranked_pairs(self) -> List[Tuple[str, float]]:
        """Rank the cumulative pool using the paper's formula-first rule."""
        formula_set = self._ranking_formula_set()
        sorted_pairs = sorted(
            self.smiles_to_similarity.items(),
            key=lambda item: (
                self.smiles_to_formula.get(item[0]) not in formula_set,
                -item[1],
                item[0],
            ),
        )
        return sorted_pairs

    def get_top_k_smiles(self, k: int) -> List[str]:
        """Get top K unique SMILES with formula priority, then similarity."""
        return [smiles for smiles, _similarity in self._ranked_pairs()[:k]]
    
    def get_ranked_smiles(self) -> List[str]:
        """Get all unique SMILES ranked by similarity (descending)."""
        return [smiles for smiles, _similarity in self._ranked_pairs()]

    def mutable_state(self) -> Dict[str, Any]:
        """Return the complete mutable state required by a later round."""
        return {
            'name': self.name,
            'all_unique_smiles': sorted(self.all_unique_smiles),
            'all_unique_ikeys': sorted(self.all_unique_ikeys),
            'smiles_to_similarity': dict(self.smiles_to_similarity),
            'smiles_to_formula': dict(self.smiles_to_formula),
        }

    def restore_mutable_state(self, value: Dict[str, Any]) -> None:
        """Restore state while keeping immutable spectrum inputs freshly loaded."""
        if value.get('name') != self.name:
            raise ValueError(
                f"resume sample mismatch: expected {self.name}, got {value.get('name')}"
            )
        self.all_unique_smiles = set(value['all_unique_smiles'])
        self.all_unique_ikeys = set(value['all_unique_ikeys'])
        self.smiles_to_similarity = {
            str(key): float(score)
            for key, score in value['smiles_to_similarity'].items()
        }
        stored_formulas = value.get('smiles_to_formula')
        if stored_formulas is None:
            self.smiles_to_formula = {
                smiles: get_molecular_formula(smiles)
                for smiles in self.all_unique_smiles
            }
        else:
            self.smiles_to_formula = {
                str(smiles): formula
                for smiles, formula in stored_formulas.items()
            }
        if set(self.smiles_to_similarity) != self.all_unique_smiles:
            raise ValueError(f'inconsistent resumed candidate state for {self.name}')
        if set(self.smiles_to_formula) != self.all_unique_smiles:
            raise ValueError(f'inconsistent resumed formula state for {self.name}')


class IcebergSampler:
    """
    ICEBERG-guided inference-time scaling sampler for DLM.
    
    This class orchestrates the full inference-time scaling pipeline:
    1. Generate molecules from scratch (round 0)
    2. Run batched ICEBERG predictions
    3. Identify hallucinated peaks and create masked inputs
    4. Refine molecules using masked generation
    5. Repeat for R rounds
    
    For efficiency, ICEBERG predictions are batched across all test samples
    and cached to avoid redundant computation.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        iceberg_config: IcebergConfig,
        scaling_config: ScalingConfig,
        results_dir: str = '/tmp/iceberg_scaling',
        token_model: Optional[Any] = None,
        token_features: Optional[List[str]] = None,
        is_ngboost: bool = False,
        sigma_lambda: float = 3.0,
        fp_bits: int = 4096,
        fp_radius: int = 2,
        masking_strategy: Optional[BaseMaskingStrategy] = None,
        masking_strategy_name: str = 'simple',
        incl_unknown_instrument: bool = True,
    ):
        """
        Initialize the ICEBERG sampler.
        
        Args:
            sampler: DLM Sampler instance
            iceberg_config: ICEBERG configuration
            scaling_config: Scaling configuration
            results_dir: Directory for intermediate ICEBERG results
            token_model: Optional token count prediction model
            token_features: Features for token model
            is_ngboost: Whether token model is NGBoost
            sigma_lambda: Variance multiplier for NGBoost sampling
            fp_bits: Fingerprint bits
            fp_radius: Fingerprint radius
            masking_strategy: Optional pre-configured masking strategy instance.
                              If None, creates one based on masking_strategy_name.
            masking_strategy_name: Name of masking strategy to use if masking_strategy
                                   is None. Options: 'simple', 'intensity_weighted'
            incl_unknown_instrument: Whether to include 'Unknown' instrument type in ICEBERG predictions
        """
        self.sampler = sampler
        self.iceberg_config = iceberg_config
        self.scaling_config = scaling_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.incl_unknown_instrument = incl_unknown_instrument
        
        # Validate configuration
        assert scaling_config.batch_size > 0, "batch_size must be positive"
        #assert scaling_config.num_unique_to_refine > 0, "num_unique_to_refine must be positive"
        #assert scaling_config.masks_per_molecule > 0, "masks_per_molecule must be positive"
        assert scaling_config.num_rounds > 0, "num_rounds must be positive"
        assert 0.0 <= scaling_config.mask_prob <= 1.0, "mask_prob must be in [0, 1]"
        
        # Ensure B >= M*K for rounds > 0
        M_times_K = scaling_config.masks_per_molecule * scaling_config.num_unique_to_refine
        assert scaling_config.batch_size >= M_times_K, (
            f"batch_size ({scaling_config.batch_size}) must be >= "
            f"masks_per_molecule * num_unique_to_refine ({M_times_K})"
        )
        
        # Token model for length prediction
        self.token_model = token_model
        self.token_features = token_features
        self.is_ngboost = is_ngboost
        self.sigma_lambda = sigma_lambda
        
        # Fingerprint settings
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        
        # ICEBERG prediction cache
        self._pred_cache = IcebergPredictionCache()
        self._last_round_proposals: Dict[str, List[Dict[str, Any]]] = {}
        self._resume_state: Optional[Dict[str, Any]] = None
        
        # Masking strategy - use provided or create from name
        if masking_strategy is not None:
            self._masking_strategy = masking_strategy
        else:
            self._masking_strategy = create_masking_strategy(
                strategy_name=masking_strategy_name,
                top_k_peaks=scaling_config.top_k_halluc_peaks,
                ppm=iceberg_config.ppm,
                real_inten_threshold=scaling_config.halluc_inten_threshold,
                sim_inten_threshold=0.05,
            )

        print(f"Using masking strategy: {type(self._masking_strategy).__name__}")

        # Tokenizer for masking
        self._tokenizer = sampler.model.tokenizer

    def _seed_sample_round(self, sample_name: str, round_idx: int) -> int:
        """Make sampling invariant to shard count, shard order, and Python hash seed."""
        payload = (
            f'{self.scaling_config.seed}\0{sample_name}\0{round_idx}'.encode('utf-8')
        )
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed % (2**63 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**63 - 1))
        return seed
    
    def _predict_token_lengths(
        self,
        formula: str,
        num_samples: int,
    ) -> Optional[List[int]]:
        """Predict token lengths for a formula."""
        if self.token_model is None or self.token_features is None:
            return None
        
        predicted_len, predicted_sigma = predict_token_count(
            self.token_model,
            self.token_features,
            formula,
            self.is_ngboost,
            self.sigma_lambda,
        )
        
        if predicted_len is None:
            return None
        
        if self.is_ngboost and predicted_sigma is not None:
            # NGBoost: sample from normal distribution
            # NGBoost ``scale`` is a standard deviation.  The paper samples
            # N(mu, lambda * sigma^2), hence numpy's ``scale`` argument is
            # sigma * sqrt(lambda), not sqrt(sigma * lambda).
            std_dev = predicted_sigma * np.sqrt(self.sigma_lambda)
            lengths = [
                max(1, int(round(np.random.normal(predicted_len, std_dev))))
                for _ in range(num_samples)
            ]
        else:
            # sklearn: sample from fixed range
            low = max(1, predicted_len - 3)
            high = predicted_len + 3
            lengths = [
                np.random.randint(low, high + 1)
                for _ in range(num_samples)
            ]
        
        return lengths
    
    def _generate_from_scratch(
        self,
        state: SampleState,
        num_samples: int,
    ) -> List[str]:
        """
        Generate molecules from scratch (fully masked).
        
        Args:
            state: Sample state with target fingerprint and formula
            num_samples: Number of samples to generate
            
        Returns:
            List of generated SMILES
        """
        if num_samples <= 0:
            return []
        
        # Check if we have candidate formulas to sample from
        if state.candidate_formulas is not None and len(state.candidate_formulas) > 0:
            # Sample formulas for each generation
            sampled_formulas = state.sample_formulas(num_samples)
            
            # Group samples by formula for efficient batching
            from collections import Counter
            formula_counts = Counter(sampled_formulas)
            
            all_samples = []
            for formula, count in formula_counts.items():
                # Predict target lengths for this formula
                target_lengths = self._predict_token_lengths(formula, count)
                
                # Generate using unified conditioning
                try:
                    samples = self.sampler.unified_conditioned_generation(
                        formula=formula,
                        fingerprint=state.target_fp,
                        num_samples=count,
                        softmax_temp=self.scaling_config.softmax_temp,
                        randomness=self.scaling_config.randomness,
                        target_lengths=target_lengths,
                        min_add_len=2,
                    )
                    all_samples.extend(samples)
                except Exception as e:
                    print(f"Warning: Generation failed for {state.name} with formula {formula}: {e}")
            
            return all_samples
        else:
            # Use fixed target formula (default behavior)
            target_lengths = self._predict_token_lengths(
                state.target_formula,
                num_samples,
            )
            
            # Generate using unified conditioning
            try:
                samples = self.sampler.unified_conditioned_generation(
                    formula=state.target_formula,
                    fingerprint=state.target_fp,
                    num_samples=num_samples,
                    softmax_temp=self.scaling_config.softmax_temp,
                    randomness=self.scaling_config.randomness,
                    target_lengths=target_lengths,
                    min_add_len=2,
                )
            except Exception as e:
                print(f"Warning: Generation failed for {state.name}: {e}")
                if os.environ.get('FRIGID_RERAISE_GENERATION_ERRORS', '0') == '1':
                    raise
                samples = []
            
            return samples
    
    def _generate_from_masked(
        self,
        masked_input_ids: torch.Tensor,
        formula: Union[str, List[str]],
        fingerprint: np.ndarray,
    ) -> List[str]:
        """
        Generate molecules from masked input.
        
        Args:
            masked_input_ids: Tensor of shape [batch, seq_len] with masked tokens
            formula: Target molecular formula (string) or list of formulas (one per batch item)
            fingerprint: Target fingerprint
            
        Returns:
            List of generated SMILES
        """
        if masked_input_ids.size(0) == 0:
            return []
        
        try:
            samples = self.sampler.generate(
                masked_input_ids,
                softmax_temp=self.scaling_config.softmax_temp,
                randomness=self.scaling_config.randomness,
                formula=formula,
                fingerprint=fingerprint,
            )
        except Exception as e:
            print(f"Warning: Masked generation failed: {e}")
            if os.environ.get('FRIGID_RERAISE_GENERATION_ERRORS', '0') == '1':
                raise
            samples = []
        
        return samples
    
    def _run_iceberg_batch(self, smiles_list, collision_energies, instrument="Unknown", ionization="[M+H]+"):
        raise NotImplementedError("A portable forward-prediction provider is required")
    
    def _run_batched_iceberg_predictions(
        self,
        sample_states: List[SampleState],
        molecules_per_sample: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, common.CompositeMassSpec]]:
        """
        Run batched ICEBERG predictions across all samples.
        
        Groups predictions by (collision_energy, instrument) pairs for efficiency.
        
        Args:
            sample_states: List of sample states
            molecules_per_sample: Dict mapping sample name -> list of SMILES
            
        Returns:
            Dict mapping sample_name -> {smiles -> pred_spec}
        """
        # Collect all unique (SMILES, instrument, ionization) triples across all samples
        smiles_instrument_ionization_triples: List[Tuple[str, str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        for state in sample_states:
            smiles_list = molecules_per_sample.get(state.name, [])
            for smi in smiles_list:
                triple = (smi, state.instrument, state.ionization)
                if triple not in seen_triples:
                    smiles_instrument_ionization_triples.append(triple)
                    seen_triples.add(triple)

        # Get uncached (CE, instrument, ionization) -> smiles_list
        ce_inst_ion_to_smiles = self._pred_cache.get_uncached_pairs_grouped(
            smiles_instrument_ionization_triples,
            self.scaling_config.collision_energies,
        )

        # Retain the pre-optimization execution shape as an explicit validation
        # control.  Production uses grouped collision energies; setting this
        # environment variable to 0 is only for same-GPU equivalence tests.
        if os.environ.get(
            'FRIGID_GROUP_ICEBERG_COLLISION_ENERGIES', '1'
        ) == '0':
            for (ce, instrument, ionization), smiles_batch in ce_inst_ion_to_smiles.items():
                if not smiles_batch:
                    continue
                print(
                    f"  [legacy-control] Running ICEBERG for {len(smiles_batch)} "
                    f"molecules at CE={ce}eV, instrument={instrument}, "
                    f"ionization={ionization}..."
                )
                pred_results = self._run_iceberg_batch(
                    smiles_batch,
                    [ce],
                    instrument=instrument,
                    ionization=ionization,
                )
                for smi, pred_spec in pred_results.items():
                    self._pred_cache.put(
                        smi,
                        ce,
                        instrument,
                        pred_spec,
                        ionization=ionization,
                    )
            ce_inst_ion_to_smiles = {}

        # A single ICEBERG invocation accepts multiple collision energies.  Group
        # by instrument/adduct so the generator and intensity checkpoints are
        # loaded once per group rather than once per collision energy.  The
        # resulting composite spectrum is split back into the original
        # nominal-CE cache keys below, preserving downstream cache semantics.
        grouped_missing: Dict[
            Tuple[str, str], Dict[str, Set[int]]
        ] = defaultdict(lambda: defaultdict(set))
        for (ce, instrument, ionization), smiles_batch in ce_inst_ion_to_smiles.items():
            for smi in smiles_batch:
                grouped_missing[(instrument, ionization)][smi].add(ce)

        for (instrument, ionization), missing_by_smiles in grouped_missing.items():
            smiles_batch = sorted(missing_by_smiles)
            collision_energies = sorted(
                {ce for values in missing_by_smiles.values() for ce in values}
            )
            if not smiles_batch or not collision_energies:
                continue

            print(
                f"  Running ICEBERG for {len(smiles_batch)} molecules at "
                f"CEs={collision_energies}, instrument={instrument}, "
                f"ionization={ionization}..."
            )
            pred_results = self._run_iceberg_batch(
                smiles_batch,
                collision_energies,
                instrument=instrument,
                ionization=ionization,
            )

            missing_by_ikey = {
                self._pred_cache.get_ikey(smi): values
                for smi, values in missing_by_smiles.items()
            }
            for smi, pred_spec in pred_results.items():
                ikey = self._pred_cache.get_ikey(smi)
                needed = missing_by_ikey.get(ikey, set())
                precursor_mass = (
                    common.mass_from_smi(smi) + common.ion2mass[ionization]
                )
                for ce in sorted(needed):
                    actual_ce = (
                        common.nce_to_ev(ce, precursor_mass)
                        if self.scaling_config.nce
                        else ce
                    )
                    actual_key = f"{float(actual_ce):.0f}"
                    if actual_key not in pred_spec.keys():
                        raise KeyError(
                            f"ICEBERG output missing CE={actual_key} for {smi}; "
                            f"available={sorted(pred_spec.keys())}"
                        )
                    single_ce_spec = common.CompositeMassSpec(
                        [pred_spec[actual_key]]
                    )
                    self._pred_cache.put(
                        smi,
                        ce,
                        instrument,
                        single_ce_spec,
                        ionization=ionization,
                    )
        
        # Build per-sample results from cache
        results: Dict[str, Dict[str, common.CompositeMassSpec]] = {}
        
        for state in sample_states:
            sample_name = state.name
            smiles_list = molecules_per_sample.get(sample_name, [])
            instrument = state.instrument
            ionization = state.ionization

            results[sample_name] = {}
            for smi in smiles_list:
                # Combine predictions across all CEs into one CompositeMassSpec
                # Collect all MassSpec objects from different collision energies
                all_mass_specs = []
                for ce in self.scaling_config.collision_energies:
                    pred_spec = self._pred_cache.get(smi, ce, instrument, ionization)
                    if pred_spec is not None:
                        # Extract MassSpec objects from the CompositeMassSpec
                        for ms in pred_spec.values():
                            all_mass_specs.append(ms)
                
                if all_mass_specs:
                    # Normalize root_canonical_smiles to avoid assertion error in CompositeMassSpec
                    # Different ICEBERG runs might return slightly different SMILES representations
                    canonical_smi = all_mass_specs[0].root_canonical_smiles
                    for ms in all_mass_specs:
                        ms.root_canonical_smiles = canonical_smi
                    
                    # Create a new CompositeMassSpec from all collected MassSpec objects
                    combined_spec = common.CompositeMassSpec(all_mass_specs)
                    results[sample_name][smi] = combined_spec
        
        return results
    
    def _create_masked_inputs_for_sample(
        self,
        state: SampleState,
        smiles_to_refine: List[str],
        pred_specs: Dict[str, common.CompositeMassSpec],
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Create masked inputs for molecules to refine.
        
        Args:
            state: Sample state with real spectrum
            smiles_to_refine: List of SMILES to refine
            pred_specs: Dict mapping SMILES -> predicted spectrum
            
        Returns:
            Tuple of (list of masked input tensors, list of source SMILES)
        """
        masked_inputs: List[torch.Tensor] = []
        source_smiles: List[str] = []
        
        for smiles in smiles_to_refine:
            pred_spec = pred_specs.get(smiles)
            if pred_spec is None:
                continue
            iceberg_smiles = pred_spec.root_canonical_smiles or smiles
            
            # Get hallucinated peaks
            try:
                halluc_peaks = self._masking_strategy.get_hallucinated_peaks(
                    state.real_specs,
                    pred_spec,
                    real_smi=iceberg_smiles,
                )
            except Exception as e:
                print(f"Warning: Failed to get hallucinated peaks for {smiles}: {e}")
                continue
            
            # Create M masked versions
            try:
                masks = self._masking_strategy.create_masked_inputs(
                    smiles=iceberg_smiles,
                    hallucinated_peaks=halluc_peaks,
                    tokenizer=self._tokenizer,
                    num_masks=self.scaling_config.masks_per_molecule,
                    mask_prob=self.scaling_config.mask_prob,
                )
                
                for mask_tensor in masks:
                    masked_inputs.append(mask_tensor)
                    source_smiles.append(smiles)
                    
            except Exception as e:
                print(f"Warning: Failed to create masked inputs for {smiles}: {e}")
                continue
        
        return masked_inputs, source_smiles
    
    def _run_single_round(
        self,
        sample_states: List[SampleState],
        round_idx: int,
        pred_specs_cache: Dict[str, Dict[str, common.CompositeMassSpec]],
        verbose: bool = False,
    ) -> Dict[str, Dict[str, common.CompositeMassSpec]]:
        """
        Run a single scaling round for all samples.
        
        The round structure is:
        - Round 0: Generate B molecules from scratch for each sample
        - Round 1+: Generate (B - M*K) from scratch + M*K refined from top K
        
        At the end of each round, we run ICEBERG on the top K molecules
        to prepare for the next round's refinement.
        
        Args:
            sample_states: List of sample states
            round_idx: Current round index (0-indexed)
            pred_specs_cache: Cached ICEBERG predictions from previous round
                             Dict mapping sample_name -> {smiles -> pred_spec}
            verbose: Whether to print detailed progress
            
        Returns:
            Updated pred_specs_cache with new ICEBERG predictions for top K molecules
        """
        B = self.scaling_config.batch_size
        K = self.scaling_config.num_unique_to_refine
        M = self.scaling_config.masks_per_molecule
        
        is_first_round = (round_idx == 0)
        
        molecules_per_sample: Dict[str, List[str]] = {}
        round_proposals: Dict[str, List[Dict[str, Any]]] = {}
        
        if is_first_round:
            # =====================================================================
            # FIRST ROUND: All B generations from scratch
            # =====================================================================
            print(f"\n  Round {round_idx + 1}: Generating {B} molecules from scratch...")
            
            for state_index, state in enumerate(
                tqdm(sample_states, desc=f"Round {round_idx + 1} Generation", disable=not verbose)
            ):
                self._seed_sample_round(state.name, round_idx)
                scratch_samples = self._generate_from_scratch(state, B)
                records = []
                for smiles in scratch_samples:
                    accepted = state.add_smiles(smiles, self.fp_bits, self.fp_radius)
                    records.append({
                        'smiles': smiles,
                        'source_type': 'scratch',
                        'source_smiles': None,
                        'accepted_unique': accepted,
                    })
                round_proposals[state.name] = records
                molecules_per_sample[state.name] = scratch_samples
                completed = state_index + 1
                if completed % 5 == 0 or completed == len(sample_states):
                    print(
                        f"  Round {round_idx + 1}: generation progress "
                        f"{completed}/{len(sample_states)} spectra",
                        flush=True,
                    )
        else:
            # =====================================================================
            # LATER ROUNDS: (B - M*K) from scratch + M*K refined
            # =====================================================================
            num_scratch = max(0, B - M * K)
            num_refined_target = M * K
            
            print(f"\n  Round {round_idx + 1}: Generating {num_scratch} from scratch + up to {num_refined_target} refined...")
            
            for state_index, state in enumerate(
                tqdm(sample_states, desc=f"Round {round_idx + 1} Generation", disable=not verbose)
            ):
                self._seed_sample_round(state.name, round_idx)
                generated = []
                proposal_sources: List[Tuple[str, Optional[str]]] = []
                
                # Part 1: Generate from scratch
                if num_scratch > 0:
                    scratch_samples = self._generate_from_scratch(state, num_scratch)
                    generated.extend(scratch_samples)
                    proposal_sources.extend(
                        [('scratch', None)] * len(scratch_samples)
                    )
                
                # Part 2: Refine top K molecules using cached ICEBERG predictions
                pred_specs = pred_specs_cache.get(state.name, {})
                top_k_smiles = state.get_top_k_smiles(K)
                
                # Create masked inputs for top K
                masked_inputs, source_smiles = self._create_masked_inputs_for_sample(
                    state,
                    top_k_smiles,
                    pred_specs,
                )
                
                if masked_inputs:
                    # Batch and pad masked inputs
                    max_len = max(t.size(-1) for t in masked_inputs)
                    padded = []
                    for t in masked_inputs:
                        if t.size(-1) < max_len:
                            pad_size = max_len - t.size(-1)
                            t = torch.nn.functional.pad(
                                t, (0, pad_size), 
                                value=self._tokenizer.pad_token_id
                            )
                        padded.append(t)
                    
                    batched = torch.cat(padded, dim=0)
                    
                    # Generate refinements
                    # When using candidate formulas, sample a formula for each masked input
                    if state.candidate_formulas is not None and len(state.candidate_formulas) > 0:
                        # Sample formulas for each refinement
                        refinement_formulas = state.sample_formulas(batched.size(0))
                        refined = self._generate_from_masked(
                            batched,
                            refinement_formulas,  # Pass list of formulas
                            state.target_fp,
                        )
                    else:
                        refined = self._generate_from_masked(
                            batched,
                            state.target_formula,
                            state.target_fp,
                        )
                    generated.extend(refined)
                    proposal_sources.extend(
                        [
                            ('refined', source_smiles[i] if i < len(source_smiles) else None)
                            for i in range(len(refined))
                        ]
                    )
                
                # Add all generated to state
                records = []
                for proposal_index, smiles in enumerate(generated):
                    accepted = state.add_smiles(smiles, self.fp_bits, self.fp_radius)
                    source_type, source_smiles_value = (
                        proposal_sources[proposal_index]
                        if proposal_index < len(proposal_sources)
                        else ('unknown', None)
                    )
                    records.append({
                        'smiles': smiles,
                        'source_type': source_type,
                        'source_smiles': source_smiles_value,
                        'accepted_unique': accepted,
                    })
                round_proposals[state.name] = records
                molecules_per_sample[state.name] = generated
                completed = state_index + 1
                if completed % 5 == 0 or completed == len(sample_states):
                    print(
                        f"  Round {round_idx + 1}: generation progress "
                        f"{completed}/{len(sample_states)} spectra",
                        flush=True,
                    )
        
        # =====================================================================
        # Run ICEBERG predictions for top K molecules (preparation for next round)
        # =====================================================================
        self._last_round_proposals = round_proposals

        if (
            round_idx + 1 == self.scaling_config.num_rounds
            or self.scaling_config.defer_next_round_iceberg
        ):
            reason = (
                "because preparation was explicitly deferred"
                if self.scaling_config.defer_next_round_iceberg
                else "because no subsequent refinement round consumes it"
            )
            print(
                f"  Round {round_idx + 1}: Skipping terminal ICEBERG prediction "
                f"{reason}."
            )
            return pred_specs_cache

        print(f"  Round {round_idx + 1}: Running ICEBERG predictions for top {K} molecules...")
        
        # Collect top K molecules for each sample
        top_k_per_sample: Dict[str, List[str]] = {}
        for state in sample_states:
            top_k_per_sample[state.name] = state.get_top_k_smiles(K)
        
        # Run batched ICEBERG predictions
        new_pred_specs_cache = self._run_batched_iceberg_predictions(
            sample_states,
            top_k_per_sample,
        )
        
        return new_pred_specs_cache
    
    def run_scaling_all_samples(
        self,
        test_samples: List[Dict[str, Any]],
        verbose: bool = False,
        round_callback: Optional[Callable[[int, List[Dict], Dict[str, float]], None]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        start_round: int = 0,
        stop_after_round: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the full scaling pipeline for all test samples.
        
        The algorithm follows "Spectrum-Error Guided Refinement":
        - Round 0: Generate B molecules from scratch
        - Round 1+: Generate (B - M*K) from scratch + M*K refined from top K
        
        ICEBERG predictions are run at the END of each round to prepare
        for the next round's refinement. This enables efficient batching.
        
        Args:
            test_samples: List of test sample dicts with keys:
                - 'name': Spectrum name
                - 'real_specs': Real CompositeMassSpec
                - 'target_fp': Target fingerprint (numpy array)
                - 'target_formula': Target molecular formula
                - 'instrument': (Optional) Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
                - 'ionization': (Optional) Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+'). Defaults to '[M+H]+'
                - 'candidate_formulas': (Optional) List of (formula, probability) tuples
            verbose: Whether to print detailed progress
            round_callback: Optional callback called after each round with
                           (round_idx, sample_states, timing_info) where:
                           - sample_states is List[Dict] with 'all_unique_smiles' and 'smiles_to_similarity' keys
                           - timing_info is Dict with 'round_time_seconds', 'cumulative_time_seconds',
                             and 'time_per_sample_seconds' keys
        
        Returns:
            List of result dicts with keys:
                - 'name': Spectrum name
                - 'generated_smiles': List of all unique SMILES (ranked by similarity)
                - 'num_unique': Number of unique molecules generated
        """
        # Initialize sample states
        sample_states: List[SampleState] = []
        
        for sample in test_samples:
            state = SampleState(
                name=sample['name'],
                real_specs=sample['real_specs'],
                target_fp=sample['target_fp'],
                target_formula=sample['target_formula'],
                instrument=sample.get('instrument', 'Unknown'),
                ionization=sample.get('ionization', '[M+H]+'),
                candidate_formulas=sample.get('candidate_formulas'),
            )
            sample_states.append(state)
        
        assert len(sample_states) > 0, "No valid test samples provided"
        
        # Run scaling rounds
        num_rounds = self.scaling_config.num_rounds
        num_samples = len(sample_states)
        
        # Cache for ICEBERG predictions (carried across rounds)
        pred_specs_cache: Dict[str, Dict[str, common.CompositeMassSpec]] = {}
        
        # Timing tracking
        cumulative_time = 0.0
        round_times: List[float] = []

        if resume_state is not None:
            if int(resume_state.get('version', -1)) != 1:
                raise ValueError('unsupported resume-state version')
            if int(resume_state['next_round']) != start_round:
                raise ValueError(
                    f"resume round mismatch: state expects {resume_state['next_round']}, "
                    f"requested {start_round}"
                )
            resumed_samples = resume_state['sample_states']
            if [value['name'] for value in resumed_samples] != [s.name for s in sample_states]:
                raise ValueError('resume sample order/name mismatch')
            for state, value in zip(sample_states, resumed_samples):
                state.restore_mutable_state(value)
            pred_specs_cache = resume_state['pred_specs_cache']
            self._pred_cache = resume_state['prediction_cache']
            cumulative_time = float(resume_state.get('cumulative_time', 0.0))
            round_times = [float(value) for value in resume_state.get('round_times', [])]
            iceberg_prepared = bool(
                resume_state.get('next_round_iceberg_prepared', True)
            )
            if start_round > 0 and not iceberg_prepared:
                print(
                    f"Preparing deferred ICEBERG cache for global round {start_round + 1} "
                    f"from frozen round-{start_round} Top-{self.scaling_config.num_unique_to_refine}...",
                    flush=True,
                )
                top_k_per_sample = {
                    state.name: state.get_top_k_smiles(
                        self.scaling_config.num_unique_to_refine
                    )
                    for state in sample_states
                }
                pred_specs_cache = self._run_batched_iceberg_predictions(
                    sample_states,
                    top_k_per_sample,
                )
        elif start_round != 0:
            raise ValueError('start_round > 0 requires resume_state')

        if stop_after_round is None:
            stop_after_round = num_rounds
        if not start_round < stop_after_round <= num_rounds:
            raise ValueError(
                f'invalid round interval [{start_round}, {stop_after_round}) '
                f'for total {num_rounds}'
            )
        
        for round_idx in range(start_round, stop_after_round):
            print(f"\n{'='*60}")
            print(f"SCALING ROUND {round_idx + 1}/{num_rounds}")
            print(f"{'='*60}")
            
            # Start timing for this round
            round_start_time = time.time()
            
            # Synchronize CUDA before starting timer for accurate GPU timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Run the round and get updated cache
            pred_specs_cache = self._run_single_round(
                sample_states, 
                round_idx, 
                pred_specs_cache,
                verbose
            )
            
            # Synchronize CUDA and record timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            round_time = time.time() - round_start_time
            cumulative_time += round_time
            round_times.append(round_time)
            
            # Report progress
            total_unique = sum(len(s.all_unique_smiles) for s in sample_states)
            avg_unique = total_unique / len(sample_states)
            print(f"  Total unique molecules: {total_unique} (avg {avg_unique:.1f} per sample)")
            print(f"  ICEBERG cache size: {len(self._pred_cache)}")
            print(f"  Round time: {round_time:.2f}s ({round_time/num_samples:.4f}s per sample)")
            print(f"  Cumulative time: {cumulative_time:.2f}s ({cumulative_time/num_samples:.4f}s per sample)")
            
            # Call round callback if provided
            if round_callback is not None:
                # Convert to dict format for callback
                state_dicts = [
                    {
                        'name': s.name,
                        'all_unique_smiles': s.all_unique_smiles,
                        'smiles_to_similarity': s.smiles_to_similarity,
                        'next_round_top_k': s.get_top_k_smiles(
                            self.scaling_config.num_unique_to_refine
                        ),
                        'round_proposals': self._last_round_proposals.get(s.name, []),
                    }
                    for s in sample_states
                ]
                # Build timing info dict
                timing_info = {
                    'round_time_seconds': round_time,
                    'cumulative_time_seconds': cumulative_time,
                    'time_per_sample_seconds': cumulative_time / num_samples,
                    'round_times': list(round_times),  # Copy of all round times so far
                }
                round_callback(round_idx, state_dicts, timing_info)

            # Persist only cache entries needed by the next round's current
            # formula-prioritized Top-K. The per-sample cache remains the
            # authoritative refinement input.
            resume_prediction_cache = IcebergPredictionCache()
            for state in sample_states:
                for smiles, combined_spec in pred_specs_cache.get(state.name, {}).items():
                    precursor_mass = (
                        common.mass_from_smi(smiles)
                        + common.ion2mass[state.ionization]
                    )
                    for ce in self.scaling_config.collision_energies:
                        actual_ce = (
                            common.nce_to_ev(ce, precursor_mass)
                            if self.scaling_config.nce
                            else ce
                        )
                        actual_key = f'{float(actual_ce):.0f}'
                        if actual_key in combined_spec.keys():
                            resume_prediction_cache.put(
                                smiles,
                                ce,
                                state.instrument,
                                common.CompositeMassSpec([combined_spec[actual_key]]),
                                ionization=state.ionization,
                            )
            self._resume_state = {
                'version': 1,
                'next_round': round_idx + 1,
                'total_rounds': num_rounds,
                'sample_states': [state.mutable_state() for state in sample_states],
                'pred_specs_cache': pred_specs_cache,
                'prediction_cache': resume_prediction_cache,
                'cumulative_time': cumulative_time,
                'round_times': list(round_times),
                'next_round_iceberg_prepared': not bool(
                    self.scaling_config.defer_next_round_iceberg
                ),
            }
        
        # Build final results
        results: List[Dict[str, Any]] = []
        
        for state in sample_states:
            ranked_smiles = state.get_ranked_smiles()
            
            results.append({
                'name': state.name,
                'generated_smiles': ranked_smiles,
                'num_unique': len(ranked_smiles),
            })
        
        return results

    def get_resume_state(self) -> Dict[str, Any]:
        """Return the state produced at the latest completed round boundary."""
        if self._resume_state is None:
            raise RuntimeError('no round has completed; resume state is unavailable')
        return self._resume_state
    
    def run_scaling_single_sample(
        self,
        name: str,
        real_specs: common.CompositeMassSpec,
        target_fp: np.ndarray,
        target_formula: str,
        instrument: str = 'Unknown',
        ionization: str = '[M+H]+',
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run scaling for a single test sample.

        This is a convenience method for processing one sample at a time.
        For efficiency, prefer run_scaling_all_samples when processing multiple samples.

        Args:
            name: Spectrum name
            real_specs: Real composite mass spectrum
            target_fp: Target fingerprint
            target_formula: Target molecular formula
            instrument: Instrument type ('Orbitrap', 'QTOF', or 'Unknown')
            ionization: Ionization/adduct type (e.g. '[M+H]+', '[M+Na]+')
            verbose: Whether to print detailed progress

        Returns:
            Result dict with generated_smiles and num_unique
        """
        test_sample = {
            'name': name,
            'real_specs': real_specs,
            'target_fp': target_fp,
            'target_formula': target_formula,
            'instrument': instrument,
            'ionization': ionization,
        }
        
        results = self.run_scaling_all_samples([test_sample], verbose)
        
        assert len(results) == 1, "Expected single result"
        return results[0]
