model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite
model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite-Chat

method=double_linear
args="6,2,5,13"

method=uniform
args="5"

python scripts/utils/modify_config.py \
    --model_path ${model_path} \
    --method ${method} \
    --args ${args}

