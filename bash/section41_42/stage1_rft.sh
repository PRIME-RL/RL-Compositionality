export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage1-rft
export MODEL_PATH=/fs-computility/prime/shared/chenweize/models/meta-llama/Llama-3.1-8B-Instruct-plain-prompt
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=data/string_task/stage1_level1/rft_data/train.parquet 
export VAL_FILES=data/string_task/stage1_level1/rft_data/test.parquet 
export BATCH_SIZE=8
export EPOCHS=2
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh