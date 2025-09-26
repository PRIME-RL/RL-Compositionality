# should be executed after bash/stage2_create_problems.sh

python3 examples/data_preprocess/split_data_into_rft_data.py \
    --data-path data/string_task/stage2_level2/train.parquet \
    --save-path data/string_task/stage2_level2_rft \
    --example-per-iter 8000 \
    --num-iter 6