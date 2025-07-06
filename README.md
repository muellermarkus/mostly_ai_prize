# Solution for the MOSTLY AI Prize (flat table)

## Overview

I implemented a simplified version of the TabularARGN [1] and use some tricks to parallelize the feature-specific models to speed up training. Second, I introduce a custom heuristic for determining which features to condition the model on. This enables a direct trade-off between overall accuracy and the relevant privacy metrics. 

The main ideas are summarized below:

- I encode all features in categorical space, similar to the TabularARGN [1],
- I check how influential each of the categorical features is using Cramer's V,
- I condition the model on the most influential features such that the resulting unique groups defined by these features do not exceed 50000.


## Replication

Make sure `uv` is installed. Then simply use `uv sync` to replicate the environment.
Next, run `main.py` using `python main.py`.
Data reports and synthetic data samples are saved in `results/`.
Training and validation losses can be accessed using tensorboard and are saved in `results/tb`.


## Relevant papers

[1] Tiwald, P., et al. (2025) TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data. https://arxiv.org/abs/2501.12012.