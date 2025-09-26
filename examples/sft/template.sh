set -x

model_path="${MODEL_PATH:-string-rft}"
project_name="${PROJECT_NAME:-string-rft}"
experiment_name="${EXPERIMENT_NAME:-llama-3.1-8b}"
train_files="${TRAIN_FILES:-}"
val_files="${VAL_FILES:-}"
bsz="${BATCH_SIZE:-128}"
max_length="${MAX_LENGTH:-3072}"
nnodes="${NNODES:-1}"
sp_size="${SP_SIZE:-1}"
epochs="${EPOCHS:-1}"
save_dir="${SAVE_DIR:-checkpoints}"


# Shift the arguments so $@ refers to the rest
shift 2

if [ $nnodes -eq 1 ]; then
    STANDALONE="--standalone"
else
    STANDALONE="--master_addr=${MLP_WORKER_0_HOST} --master_port=${MLP_WORKER_0_PORT} --node_rank=${MLP_ROLE_INDEX}"
fi

torchrun ${STANDALONE} --nnodes=${nnodes} --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=${max_length} \
    data.truncation=right \
    optim.lr=2e-5 \
    data.train_batch_size=${bsz} \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=${model_path} \
    trainer.default_hdfs_dir=${save_dir} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=['console'] \
    trainer.total_epochs=${epochs} \
    ulysses_sequence_parallel_size=${sp_size} \
    use_remove_padding=true
