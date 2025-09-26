set -x

project_name=string-sft
experiment_name=llama-3.1-8b

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/string_manipulation_sft_merged/train_balanced.parquet \
    data.val_files=data/string_manipulation_sft/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=3072 \
    optim.lr=2e-5 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=/fs-computility/prime/shared/chenweize/models/meta-llama/Llama-3.1-8B-Instruct-plain-prompt \
    trainer.default_hdfs_dir=checkpoints/${project_name}/$experiment_name \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
