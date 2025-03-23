import argparse
import json
import os
import numpy as np
import time
from datetime import datetime

log_path='/root/data/operation_log.json'
MIN_INTERVAL = 30

def wait_for_time_gap():
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
            # if len(log)<=0:
            #     print('ok')
            #     return
            last_time = datetime.fromisoformat(log['last_modified_time'])
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < MIN_INTERVAL:
                # wait_time = MIN_INTERVAL - elapsed
                wait_time = MIN_INTERVAL
                print(f"waiting {wait_time:.1f} 秒...")
                time.sleep(wait_time)

def write_log(info):
    log = {
        'last_modified_time': datetime.now().isoformat(),
        'modified_by': info
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def reset_config(args):
    if 'json' not in args.model_path:
        model_path = os.path.join(args.model_path, 'config.json')
    else:
        model_path = args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"file {model_path} doesn't exist!")

    with open(model_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    with open(model_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def parse_int_list(value):
    return [int(i.strip()) for i in value.split('_')]

def linear_distribution(vl,vr,num):
    return [vl+round((vr-vl)/num*i) for i in range(num)]

def get_topk_list(args, num_hidden_layers):
    # num_hidden_layers -= 1
    
    if args.args is not None and isinstance(args.args, list)==False:
        args_list=parse_int_list(args.args)
    
    if args.method=='uniform' or (args.method=='auto' and len(args_list)==1):
        assert len(args_list)==1
        topk_list = linear_distribution(args_list[0], args_list[0], num_hidden_layers)
        
    elif args.method=='linear' or (args.method=='auto' and len(args_list)==2):
        assert len(args_list)==2
        topk_list = linear_distribution(args_list[0], args_list[1], num_hidden_layers)
        
    elif args.method=='double_linear' or (args.method=='auto' and len(args_list)==4):
        assert len(args_list)==4
        topk_list = linear_distribution(args_list[0], args_list[1], args_list[3])
        topk_list += linear_distribution(args_list[1], args_list[2], num_hidden_layers-args_list[3])
        
    else:
        raise NotImplementedError
    assert len(topk_list)==num_hidden_layers
    topk_list.insert(0,-1)
    return topk_list

def read_json_and_print_topk(args):
    if 'json' not in args.model_path:
        model_path = os.path.join(args.model_path, 'config.json')
        # model_path = os.path.join(args.model_path, 'config_copy.json')
    else:
        model_path = args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"file {model_path} doesn't exist!")

    wait_for_time_gap()

    with open(model_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if 'topk_list' not in data:
        raise KeyError("topk_list hasn't been initialized")

    data['topk_list'] = get_topk_list(args, data['num_hidden_layers']-1)
    
    print(model_path)
    print(len(data['topk_list']))
    print(data['topk_list'])
    
    with open(model_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    
    write_log(args.args)


if __name__ == "__main__":
    choices=['reset','uniform','linear','double_linear','auto',]
    
    parser = argparse.ArgumentParser(description="Modify the config file.")
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--method', required=True, choices=choices, help=f'choices = {choices}')
    parser.add_argument('--args')
    args = parser.parse_args()
    if args.method=='reset':
        reset_config(args)
    try:
        read_json_and_print_topk(args)
    except Exception as e:
        print(f"[ERROR] {e}")
