export NNODES=1
export DATA_PATH=data/string_task/stage1_level1/train.parquet
export N_SAMPLES=10
export SAVE_PATH=data/string_task/stage1_level1/rollout.parquet
export MODEL_PATH=../models/meta-llama/Llama-3.1-8B-Instruct-plain-prompt
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192
export RFT_DATA_SAVE_PATH=data/string_task/stage1_level1/rft_data

bash examples/generation/run_string.sh

python3 examples/data_preprocess/string_manipulation_sft.py \
    --gen_path ${SAVE_PATH} \
    --data_path ${DATA_PATH} \
    --save_path ${RFT_DATA_SAVE_PATH} \
    --val_size 1 \
    --max_correct_ratio 1.0 \
    --no_remove_context
