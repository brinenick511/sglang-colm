import argparse
import json
import os
import numpy as np

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
    return [int(i.strip()) for i in value.split(',')]

def linear_distribution(vl,vr,num):
    return [vl+round((vr-vl)/num*i) for i in range(num)]

def get_topk_list(args, num_hidden_layers):
    # num_hidden_layers -= 1
    
    if args.args is not None and isinstance(args.args, list)==False:
        args_list=parse_int_list(args.args)
    
    if args.method=='uniform':
        assert len(args_list)==1
        topk_list = linear_distribution(args_list[0], args_list[0], num_hidden_layers)
        
    elif args.method=='linear':
        assert len(args_list)==2
        topk_list = linear_distribution(args_list[0], args_list[1], num_hidden_layers)
        
    elif args.method=='double_linear':
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


if __name__ == "__main__":
    choices=['reset','uniform','linear','double_linear']
    
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
