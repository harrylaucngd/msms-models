# References and upstream training

Cite the original methods as well as this integration when using their outputs.
Training is provided by the upstream projects; this repository focuses on
inference, preparation and consistent evaluation.

| Method | Paper | Upstream training / source |
|---|---|---|
| MIST | Goldman et al., [Annotating metabolite mass spectra with domain-inspired chemical formula transformers](https://doi.org/10.1038/s42256-023-00708-3) | [Original MIST](https://github.com/samgoldman97/mist); [modern DDP and optional synthetic training](https://github.com/coleygroup/MIST/blob/main/docs/training.md) |
| MIST + MolForge | Neo et al., [One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](https://arxiv.org/abs/2508.04180) | [MolForge decoder](https://github.com/knu-lcbc/MolForge); [MIST–MolForge integration](https://github.com/harrylaucngd/MIST-MolForge) |
| MS-GPT-base / MS-GPT | [Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model](https://arxiv.org/abs/2607.23607) | [VIKI623/MS-GPT](https://github.com/VIKI623/MS-GPT); published SAFE weights and dataset-specific adapters; use the matching SAFE tokenizer |
| FRIGID | [Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time](https://arxiv.org/abs/2604.16648) | [coleygroup/FRIGID](https://github.com/coleygroup/FRIGID), DLM training and formula-length modeling |
| FOAM | [Generative structural elucidation from mass spectra as an iterative optimization problem](https://arxiv.org/abs/2602.07709) | [coleygroup/foam](https://github.com/coleygroup/foam); optimization uses pretrained spectral models |
| Nearest neighbor | [Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions](https://arxiv.org/abs/2610.02249) | [serenaklm/ML_MS_analysis](https://github.com/serenaklm/ML_MS_analysis); `fit-nn` exposes train/validation fitting of the sub-formula-only variant |
| ICEBERG | [Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC12573212/) | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) |
| GLACIER | [Rethinking Mass Spectrum Prediction as an Object Detection Problem](https://arxiv.org/abs/2606.29161) | [coleygroup/ms-pred](https://github.com/coleygroup/ms-pred); [checkpoint compatibility](https://github.com/coleygroup/MIST/blob/main/docs/glacier.md) |

The original MolForge decoder is described by Ucak, Ashyrmamatov and Lee (2023),
[Reconstruction of lossless molecular representations from fingerprints](https://doi.org/10.1186/s13321-023-00693-0).
Its [upstream repository](https://github.com/knu-lcbc/MolForge) supplies the license. Dataset and metric citations should follow the
MassSpecGym benchmark and the evaluation protocol being reported. Retain model
and dataset versions in any published results.
