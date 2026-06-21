## Data & Reproducibility

This repository contains all code necessary to reproduce the figures in Condruz et al. (2026). The precomputed data required to run the figure notebooks is archived on Zenodo:

> **[Dataset] PATNs: data for Condruz 2026**  
> [https://doi.org/10.5281/zenodo.20773396](https://doi.org/10.5281/zenodo.20773396)

Download `data.zip` from Zenodo and unzip it in the root of this repository:

```bash
unzip data.zip
```

## The PATNs Algorithm

Beyond the figure notebooks, this repository includes a standalone implementation of the **PATNs (Periodic And Transient Networks) algorithm** — a method for constructing recurrent neural network connectivity matrices with hand-crafted eigenspectra, Dale's law compliance, and controlled non-normality.

The algorithm lives in `algorithm/` and comes with a self-contained manual. To get started, open `PATNs_demo.ipynb` or read `PATNs_manual.pdf` for a full description of parameters and usage.
