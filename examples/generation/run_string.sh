set -x

DATA_PATH=${DATA_PATH:-"/fs-computility/prime/shared/chenweize/synthetic-rl/data/synthetic_data/synthetic_string_manipulation_induction_with_testcase_json/test.parquet"}
SAVE_PATH=${SAVE_PATH:-"/fs-computility/prime/shared/chenweize/results/synthetic_string_manipulation_induction_with_testcase_json/step200_test.parquet"}
MODEL_PATH=${MODEL_PATH:-"/fs-computility/prime/shared/chenweize/synthetic-rl/checkpoints/string_induction/llama3.1_8b_induction_bsz16_prompt1024_response8192_mbsz16_n16_withtestcase_json/global_step_200/actor/huggingface"}
N_SAMPLES=${N_SAMPLES:-"1"}
TEMPERATURE=${TEMPERATURE:-"0"}
PROMPT_LENGTH=${PROMPT_LENGTH:-"1024"}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-"8192"}
NNODES=${NNODES:-"1"}

python3 -m verl.trainer.main_generation \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.n_samples=$N_SAMPLES \
    data.output_path=$SAVE_PATH \
    data.batch_size=100000000 \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.name=vllm \
    rollout.temperature=$TEMPERATURE \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.prompt_length=$PROMPT_LENGTH \
    rollout.response_length=$RESPONSE_LENGTH \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
