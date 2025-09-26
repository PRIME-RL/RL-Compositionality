export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage2-rft
export MODEL_PATH=checkpoints/string-task/stage1-rft-iter1
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=data/string_task/stage2_level2_rft/0.parquet 
export VAL_FILES=data/string_task/stage2_level1to8/test.parquet 
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
