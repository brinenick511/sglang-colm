
request_rate=6
num_prompt=$((300*${request_rate}))

python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --dataset-path /new_data/yanghq/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --random-range-ratio 1 \
    --num-prompt ${num_prompt} \
    --request-rate ${request_rate} \
    --random-input 1024 \
    --random-output 1024  \
    --output-file /new_data/yanghq/data/sglang/test.jsonl \
    --host 127.0.0.1 \
    --port 30000 \

