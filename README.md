# conformal-fairness

## Environment export from conda

```bash
conda env export | grep -v "name" | grep -v "prefix" > environment.yml
```

## Directory Structure

- `conformal_fairness` contains all the code for the CF Framework logic

- `datasets` contains raw data for Credit and Pokec-{n, z}. For ACS datasets, they are downloaded through the code

- `scripts` contains SLURM batch scripts used to run the experiments

- `analysis` contains noteboks for generating figures and scripts to pull from W&B

- `configs` contains configs for the different experiments

Remaining python files are used to run hyperparameter tuning and conformal prediction.
