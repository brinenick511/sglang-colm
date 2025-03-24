model_path=${HOME}/models/deepseek-ai/DeepSeek-V2-Lite

gpuid=0

method=auto
args_list=(2_2 2_4 2_6 2_8 4_2 4_4 4_6 4_8 6_2 6_4 6_6 6_8 8_2 8_4 8_6 8_8 )

for args in ${args_list[@]}
do
    echo ${args}
    # echo ${model_path}

    python scripts/utils/modify_config.py \
        --model_path ${model_path} \
        --method ${method} \
        --args ${args}

    tasks=arc_challenge,arc_easy,boolq,openbookqa,rte,winogrande

    model_args=pretrained=${model_path},trust_remote_code=True
    model_args=${model_args},enable_p2p_check=True,disable_cuda_graph=True
    model_args=${model_args},tp_size=8,kv_cache_dtype=fp8_e5m2
    model_args=${model_args},mem_fraction_static=0.75

    HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=$gpuid \
    lm_eval --model sglang \
        --model_args ${model_args} \
        --tasks ${tasks} \
        --batch_size 32 \
        --trust_remote_code \
        --num_fewshot 0 \
        --output_path ${HOME}/data/lme/${method}/${args}.json
        # --limit 256 \
    
    sleep 1

done