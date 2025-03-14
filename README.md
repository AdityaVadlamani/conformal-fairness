# Conformal Fairness

This branch contains the latest version of the code for the Conformal Fairness Framework introduced in the ICLR 2025 "A Generic Framework for Conformal Fairness".

The Conformal Fairness Framework is open-source with the Apache 2.0 license.

### Branches

- [iclr2025](https://github.com/AdityaVadlamani/conformal-fairness/tree/iclr2025): This branch contains everything needed to reproduce the results in the [ICRL 2025 paper](https://openreview.net/pdf?id=xiQNfYl33p)

## Environment export from conda

```bash
conda env export | grep -v "name" | grep -v "prefix" > environment.yml
```

## Directory Structure

- `conformal_fairness` contains all the code for the CF Framework logic

- `datasets` contains raw data for Credit and Pokec-{n, z}. For ACS datasets, they are downloaded through the code

- `scripts` contains SLURM batch scripts used to run the experiments

- `analysis` contains notebooks for generating figures and scripts to pull from W&B

- `configs` contains configs for the different experiments

- `BatchGCP` contains the code used for the BatchGCP experiments

The remaining Python files are used to run hyperparameter tuning and conformal prediction.
