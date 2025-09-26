set -x

export MODEL_PATH=checkpoints/string-task/stage1-rft
export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage2-rl-level1
export TRAIN_FILES="['data/string_task/stage2_level1/train.parquet']"
export VAL_FILES="['data/string_task/stage2_level1to8/test.parquet']"
export NNODES=1
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh