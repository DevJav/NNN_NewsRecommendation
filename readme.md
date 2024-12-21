# NNN: THE NEWS NEURAL NETWORK FOR CLICK RECOMMENDATION

### s233351 Clara Dávila Duarte, s233150 Amanda Jiménez, s234032 Gaia Casotto, s233671 Javier Moreno Prieto

## Introduction

This repository contains the code used for the project of the course "Deep Learning".

## Content

The code has as starting point the implementation of the NRMS model provided in the [ebnerd's repository](https://github.com/ebanalyse/ebnerd-benchmark/tree/main). The setup and dependecies are therefore the same as the ones in the original repository and they should be installed as instructed in its README file.

## Testing the model

We provide the files `nrms_ebnerd.py` and `main.ipynb` to test the model. The first one contains the implementation of the NRMS model that we have been using as it is easier to run with the HPC environment. The second one is a Jupyter Notebook that contains the same code as the first file but in a more readable format and with some results, but it may fail due to memory issues.

There are some important parametes that can be modified in the code:

- `SUBSAMPLE_DATASET`: if set to `True`, the model will use a subsample of the dataset to train the model. If set to `False`, the model will use the whole dataset.
- `TEST`: if set to `True`, the model will run a test with a small dataset to check if everything is working. If set to `False`, the model will run with the whole dataset. Only available in the python file.
- `PATH`: the path to the dataset. It should be set to the path where the dataset folder is stored.
- `USE_TIMESTAMPS`: **ADDITIONAL STEPS ARE NEEDED BEFORE BEING ABLE TO SET THIS PARAMETER TO TRUE**. If set to `True`, the model will use the timestamps of when the articles were clicked to train the model. If set to `False`, the model will not use the timestamps as in the original implementation.

### Running the code

To run the code, you can use the following command:

```bash
python nrms.py
```

### Including timestamps

To include the timestamps, the ebnerd library has been modified. Two files have been modified and are included in the repository: `nrms.py` and `dataloader.py`. To be able to run the model with timestamps, the original files should be replaced by the ones in the repository.

If you want to go back to the original implementation, you can replace the files with the ones in the original repository.

## Results

Validation results will be printed in the console.

If TEST is set to `True`, the code will generate a submission file for codabench.
