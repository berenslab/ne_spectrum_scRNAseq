# Neighbor embedding spectrum for single-cell data

This repository contains scripts and notebooks to reproduce the experiments in
*Exploring structure in single-cell data with the neighbor embedding spectrum* ([bioarxiv]())

It depends on the `ne-spectrum` package, which computes neighbor embedding spectra.

<p align="center"><img width="800" alt="NE spectrum on Kanton et al. data" src="/figures/main_fig_tsne.png">


Neighbor embedding spectrum on developmental human brain organoid data from [Kanton et al. 2019](https://www.nature.com/articles/s41586-019-1654-9)


![](https://github.com/ne_spectrum_scRNAseq/figures/human-409b2_tsne_seed_2_n_slides_60_min_spec_0.85_max_spec_30.0.gif)

Higher attraction improves the global structure as measured by Spearman distance correlation. Higher repulsion improves
local structure as measured by kNN recall.


<p align="center"><img width="400" alt="Global and local metric along the spectrum" src="/figures/local_global_metrics_tsne_subsample_0.png">


# Installation

Create and activate the conda environment
```
conda env create -f environment.yml
conda activate ne_spectrum_scRNAseq
```

# Usage
To reproduce the data for the figures in the main paper, run
```
python scripts/compute_embds.py --spectrum_via tsne
python scripts/compute_metrics.py --spectrum_via tsne
```

To reproduce the data for the figures in the supplementary, run
```
python scripts/compute_embds.py --spectrum_via cne
python scripts/compute_metrics.py --spectrum_via cne
```

Then run the notebooks in `notebooks/` to generate the figures and the videos.