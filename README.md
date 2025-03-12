# conformal-fairness

Code and data for the ICLR 2025 paper "A Generic Framework for Conformal Fairness". This branch contains everything to reproduce the results in the paper and **will not** be updated further beyond README changes.

NOTE: Any updates to the conformal-fairness codebase (e.g., refactoring or new features) will be reflected in the main branch or other specific branches.


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
