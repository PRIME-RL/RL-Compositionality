# Section 4.4 Pipelines — Pass@k Evaluation

This folder provides the script used to collect responses for pass@k analyses in Section 4.4.

## Script Overview

| Script | Description | Outputs |
| --- | --- | --- |
| `passk.sh` | Generates `N_SAMPLES=1000` rollouts for every prompt in the evaluation split and saves them for offline pass@k computation. | `results/stage2_rl_level1/all.parquet` |

## Usage

1. Set `MODEL_PATH` to the checkpoint you want to evaluate.
2. Optionally adjust `DATA_PATH`, `SAVE_PATH`, `N_SAMPLES`, or sampling hyperparameters (temperature, max lengths). The default configuration evaluates the Level-1 RL checkpoint on the held-out Level-1–8 mixture.

