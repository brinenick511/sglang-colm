
request_rate_list=(1 2 4 8 )

for request_rate in ${request_rate_list[@]}
do
    num_prompt=$((300*${request_rate}))
    echo ${request_rate}
    echo ${num_prompt}

    dataset_path=/root/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
    output_path=/root/data/sglang-benchmark/tp2.jsonl

    python3 -m sglang.bench_serving \
        --backend sglang \
        --dataset-name random \
        --dataset-path ${dataset_path} \
        --random-range-ratio 1 \
        --num-prompt ${num_prompt} \
        --request-rate ${request_rate} \
        --random-input 1024 \
        --random-output 1024  \
        --output-file ${output_path} \
        --host 127.0.0.1 \
        --port 30000

done