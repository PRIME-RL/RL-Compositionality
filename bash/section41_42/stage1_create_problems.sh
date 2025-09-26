python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage1_level1/train.parquet \
    --stage 1 \
    --split train \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage1_level1/test.parquet \
    --stage 1 \
    --split test \
    --min_level 1 \
    --max_level 1 \
    --data_num 128