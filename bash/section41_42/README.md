# Sections 4.1 & 4.2 Pipelines

This directory contains the bash entrypoints used to reproduce the Stage 1 (atomic skills) and Stage 2 (compositional skills) experiments from Sections 4.1 and 4.2 of the paper.

All commands assume that you are running from the repository root and that `meta-llama/Llama-3.1-8B-Instruct-plain-prompt` (or an equivalent checkpoint) is available locally. Update the `MODEL_PATH` variables to match your environment.

## Stage 1 — Atomic Skill Acquisition

| Script | Purpose | Key Outputs |
| --- | --- | --- |
| `stage1_create_problems.sh` | Generate atomic string transformation datasets for Stage 1. | `data/string_task/stage1_level1/{train,test}.parquet` |
| `stage1_create_train_data.sh` | Sample model rollouts conditioned on full function definitions and convert them into RFT splits. | `data/string_task/stage1_level1/rollout.parquet`, `data/string_task/stage1_level1/rft_data/{train,test}.parquet` |
| `stage1_rft.sh` | Fine-tune the base model with Stage 1 RFT data. | `checkpoints/string-task/stage1-rft/` |

Run the scripts sequentially to obtain the Stage 1 checkpoint.

## Stage 2 — Compositional Skill Learning

### Data Preparation

| Script | Purpose | Key Outputs |
| --- | --- | --- |
| `stage2_create_problems.sh` | Create Level-1, Level-2, and mixed Level-1+2 compositional datasets along with the held-out evaluation split. | `data/string_task/stage2_*/` Parquet files |
| `stage2_rft_create_problems.sh` | Chunk the Level-2 dataset into smaller Parquet files for iterative RFT. | `data/string_task/stage2_level2_rft/*.parquet` |
| `stage2_rft_create_train_data.sh` | Collect rollouts from the Stage 2 dataset and package them into RFT splits. | `data/string_task/stage2_rft_level2/{rollout.parquet,rft_data/}` |

### Training Recipes

| Script | Curriculum |
| --- | --- | 
| `stage2_rl_level1.sh` | RL on Level-1 (atomic) prompts only. | 
| `stage2_rl_level2.sh` | RL on Level-2 compositions only. | 
| `stage2_rl_level1to2.sh` | RL on a Level-1 + Level-2 mixture. 
| `stage2_rft_train.sh` | RFT baseline using Stage 2 data. | 

All RL scripts expect the Stage 1 checkpoint at `checkpoints/string-task/stage1-rft/`. Adjust `MODEL_PATH`, `TRAIN_FILES`, and `VAL_FILES` to experiment with different checkpoints or datasets.

