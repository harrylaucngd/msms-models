# References and upstream training

Cite the original methods as well as the paper when using their outputs. Methods
trained by upstream projects are integrated for inference; the authors' own
experiments (nearest-neighbour index, shortcut models) include training code here.

| Method | Paper | Upstream training / source |
|---|---|---|
| MIST | Goldman et al., [Annotating metabolite mass spectra with domain-inspired chemical formula transformers](https://doi.org/10.1038/s42256-023-00708-3) | [Original MIST](https://github.com/samgoldman97/mist); [modern DDP and optional synthetic training](https://github.com/coleygroup/MIST/blob/main/docs/training.md) |
| MIST + MolForge | Neo et al., [One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](https://arxiv.org/abs/2508.04180) | [MolForge decoder](https://github.com/knu-lcbc/MolForge); [MIST–MolForge integration](https://github.com/harrylaucngd/MIST-MolForge) |
| MS-GPT (decoder, density-band querying) | [Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model](https://arxiv.org/abs/2607.23607) | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT); published SAFE decoder and tokenizer; here conditioned on ScaledMIST posteriors only |
| FRIGID | [Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time](https://arxiv.org/abs/2604.16648) | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), DLM training and formula-length modeling |
| FOAM | [Generative structural elucidation from mass spectra as an iterative optimization problem](https://arxiv.org/abs/2602.07709) | [coleygroup/foam](https://github.com/coleygroup/foam); optimization uses pretrained spectral models |
| Nearest neighbor | [Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions](https://arxiv.org/abs/2610.02249) | [serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis); `fit-nn` exposes train/validation fitting of the sub-formula-only variant |
| ICEBERG | [Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC12573212/) | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) |
| GLACIER | [Rethinking Mass Spectrum Prediction as an Object Detection Problem](https://arxiv.org/abs/2606.29161) | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred); [checkpoint compatibility](https://github.com/coleygroup/MIST/blob/main/docs/glacier.md) |
| DreaMS + ChemBERTa alignment (SpecBridge) | SpecBridge (Hassoun Lab); DreaMS: Bushuiev et al., [Self-supervised learning of molecular representations from millions of tandem mass spectra](https://doi.org/10.1038/s41587-025-02663-3); ChemBERTa-2: Ahmad et al. (2022) | [HassounLab/SpecBridge](https://github.com/HassounLab/SpecBridge); [DreaMS](https://github.com/pluskal-lab/DreaMS); training in `models/retrieval/dreams_chemberta/train.py` |
| Chemformer two-stage retrieval | Generative two-stage retrieval, [arXiv:2511.06259](https://arxiv.org/abs/2511.06259); Chemformer: Irwin et al. (2022) | authors' reimplementation; [MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer); training in `models/retrieval/chemformer_two_stage/train.py` |
| ChemBERTa SMILES classifier, PubChem order, chiral atoms | paper Appendix A.4 | [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM); [PubChem PUG-REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) |
| MIST vDiffMS (diagnostic) | Bohde et al., [DiffMS](https://arxiv.org/abs/2502.09571) | [coleygroup/DiffMS](https://github.com/coleygroup/DiffMS) |
