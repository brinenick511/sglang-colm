model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite
gpuid=8,9

model_path=/new_data/yanghq/models/deepseek-ai/DeepSeek-V2-Lite-Chat
gpuid=6,7

model_path="${HOME}/$(echo "$model_path" | sed 's|^.*models/|models/|')"

method=linear

args_list=(2,2 2,3 2,4 2,5 2,6 3,2 3,3 3,4 3,5 3,6 4,2 4,3 4,4 4,5 4,6 5,2 5,3 5,4 5,5 5,6 6,2 6,3 6,4 6,5 6,6 )

for args in ${args_list[@]}
do
    echo ${args}
    # echo ${model_path}

    python scripts/utils/modify_config.py \
        --model_path ${model_path} \
        --method ${method} \
        --args ${args}

    ######
    ######

    tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande

    model_args=pretrained=${model_path},trust_remote_code=True
    model_args=${model_args},tp_size=2
    model_args=${model_args},enable_p2p_check=True,disable_cuda_graph=True
    model_args=${model_args},mem_fraction_static=0.68,chunked_prefill_size=4096
    # model_args=${model_args},

    # HF_DATASETS_TRUST_REMOTE_CODE=True \
    CUDA_VISIBLE_DEVICES=$gpuid \
    lm_eval --model sglang \
        --model_args ${model_args} \
        --tasks ${tasks} \
        --batch_size 128 \
        --trust_remote_code \
        --num_fewshot 0 \
        --output_path ${HOME}/data/lme/ \
        # --limit 256 \
    
    sleep 1
done