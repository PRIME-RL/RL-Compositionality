# Section 4.3 Pipelines — Countdown Transfer

These scripts reproduce the cross-task transfer experiments described in Section 4.3. They prepare Countdown arithmetic datasets, collect Stage 1 Countdown RFT data, merge the data with Level 1 string task data for Stage 1 RFT training.

## Scripts

| Script | Description | Outputs |
| --- | --- | --- |
| `create_countdown_data.sh` | Generates synthetic Countdown prompts and reference solutions using `examples/data_preprocess/reasoning_gym_countdown.py`. | `data/reasoning_gym/countdown/*.parquet` |
| `stage1_collect_train_data.sh` | Samples rollouts from the base model on Countdown prompts, converts them into RFT-ready splits, and merges them with Stage 1 string-task rollouts. | `data/reasoning_gym/countdown/stage1/rollout.parquet`, `data/string_countdown_task/stage1_rft_data/{train,test}.parquet` |
