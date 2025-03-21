model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite

method=double_linear

args="6,2,5,13"

python scripts/utils/modify_config.py \
    --model_path ${model_path} \
    --method ${method} \
    --args ${args}


