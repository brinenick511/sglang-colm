model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite
gpuid=0

# model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite-Chat
# gpuid=6,7

model_path="${HOME}/$(echo "$model_path" | sed 's|^.*models/|models/|')"

method=uniform
args="2"

python scripts/utils/modify_config.py \
    --model_path ${model_path} \
    --method ${method} \
    --args ${args}


tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande
# tasks=arc_easy

model_args=pretrained=${model_path},trust_remote_code=True
# model_args=${model_args},tp_size=2,chunked_prefill_size=4096
model_args=${model_args},enable_p2p_check=True,disable_cuda_graph=True
model_args=${model_args},mem_fraction_static=0.7
# model_args=${model_args},

# HF_DATASETS_TRUST_REMOTE_CODE=True \
HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=$gpuid \
lm_eval --model sglang \
    --model_args ${model_args} \
    --tasks ${tasks} \
    --batch_size 32 \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path ${HOME}/data/lme/ \
    --limit 64 \

