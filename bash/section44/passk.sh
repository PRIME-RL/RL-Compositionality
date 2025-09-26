set -x

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

export DATA_PATH=data/string_task/stage2_level1to8/test.parquet
export SAVE_PATH=results/stage2_rl_level1/all.parquet
export MODEL_PATH=checkpoints/string-task/stage2-rl-level1
export N_SAMPLES=1000
export TEMPERATURE=1.0
export NNODES=1

bash examples/generation/run_string_deduction.sh