python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage2_level1to2/train.parquet \
    --stage 2 \
    --split train \
    --min_level 1 \
    --max_level 2 \
    --data_num 50000

python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage2_level1/train.parquet \
    --stage 2 \
    --split train \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage2_level2/train.parquet \
    --stage 2 \
    --split train \
    --min_level 2 \
    --max_level 2 \
    --data_num 50000

python3 examples/data_preprocess/string_data.py \
    --save_path data/string_task/stage2_level1to8/test.parquet \
    --stage 2 \
    --split test \
    --min_level 1 \
    --max_level 8 \
    --data_num 2048