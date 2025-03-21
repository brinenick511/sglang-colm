model_path=/new_data/yanghq/models/moonshotai/Moonlight-16B-A3B-Instruct
model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite-Chat
model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite

gpuid=0,1

model_path="${HOME}/$(echo "$model_path" | sed 's|^.*models/|models/|')"




# NCCL_P2P_DISABLE=1 \
# NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=$gpuid \
python -m sglang.launch_server \
    --model $model_path \
    --trust-remote-code \
    --tp 2 \
    --mem-fraction-static 0.9 \
    --host 127.0.0.1 \
    --port 30000 \
    --enable-p2p-check \
    --enable-torch-compile \
    --chunked-prefill-size 4096 \

