import sys
import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd

try:
    import wandb
except Exception as e:
    pass

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)


# rename some variable names
def load_custom(algorithm, path):
    state = torch.load(path)
    srm_count = 0
    srm_inner_count = 0
    ff_count = 1
    ff_inner_count = 0

    if False:  # debugging mode
        for key in state['algorithm'].keys():
            if not ('vision' in key):
                print(key)

        print('===============================================')

        for name, _ in algorithm.model.named_parameters():
            if not ('vision' in name):
                print(name)

    ln_w_count = 0
    ln_b_count = 0
    ckpt = state['algorithm']
    model_keys = ckpt.copy().keys()
    for name in model_keys:
        if 'layer_norm.weight' in name:
            new_id = ln_w_count // 2
            if ln_w_count % 2 == 0:  # srm
                new_name = name.replace(f'mem_layers.{ln_w_count}',
                                        f'mem_layers.srm_layers.{new_id}')
            else:
                new_name = name.replace(f'mem_layers.{ln_w_count}',
                                        f'mem_layers.ff_layers.{new_id}')
            print(f'[custom loader] Rename: {name} --> {new_name}')
            ckpt[new_name] = ckpt[name]
            del ckpt[name]
            ln_w_count += 1

        if 'layer_norm.bias' in name:
            new_id = ln_b_count // 2
            if ln_b_count % 2 == 0:  # srm
                new_name = name.replace(f'mem_layers.{ln_b_count}',
                                        f'mem_layers.srm_layers.{new_id}')
            else:
                new_name = name.replace(f'mem_layers.{ln_b_count}',
                                        f'mem_layers.ff_layers.{new_id}')
            print(f'[custom loader] Rename: {name} --> {new_name}')
            ckpt[new_name] = ckpt[name]
            del ckpt[name]
            ln_b_count += 1

        if any(x in name for x in ['W_y', 'W_q', 'W_k', 'w_b', 'out_linear']):
            new_id = srm_count // 2
            # from:
            # model.mem_layers.0.W_y
            # model.mem_layers.0.W_q
            # model.mem_layers.0.W_k
            # model.mem_layers.0.w_b
            # model.mem_layers.0.out_linear.weight
            # to:
            # model.mem_layers.srm_layers.0.W_y
            # model.mem_layers.srm_layers.0.W_q
            # model.mem_layers.srm_layers.0.W_k
            # model.mem_layers.srm_layers.0.w_b
            # model.mem_layers.srm_layers.0.out_linear.weight
            new_name = name.replace(f'mem_layers.{srm_count}',
                                    f'mem_layers.srm_layers.{new_id}')
            print(f'[custom loader] Rename: {name} --> {new_name}')
            ckpt[new_name] = ckpt[name]
            del ckpt[name]
            srm_inner_count += 1
            if srm_inner_count == 5:
                srm_count = srm_count + 2

        if 'ff_layers' in name and any(x in name for x in ['weight', 'bias']):
            new_id = ff_count // 2
            # from:
            # model.mem_layers.1.ff_layers.0.weight
            # model.mem_layers.1.ff_layers.0.bias
            # model.mem_layers.1.ff_layers.3.weight
            # model.mem_layers.1.ff_layers.3.bias
            # to:
            # model.mem_layers.ff_layers.0.ff_layers.0.weight
            # model.mem_layers.ff_layers.0.ff_layers.0.bias
            # model.mem_layers.ff_layers.0.ff_layers.3.weight
            # model.mem_layers.ff_layers.0.ff_layers.3.bias
            new_name = name.replace(
                f"{ff_count}.ff_layers", f"ff_layers.{new_id}.ff_layers")
            print(f'[custom loader] Rename: {name} --> {new_name}')
            ckpt[new_name] = ckpt[name]
            del ckpt[name]
            ff_inner_count += 1
            if ff_inner_count == 4:
                ff_count = ff_count + 2

    # print('dict ================')
    # for key in ckpt.keys():
    #     if not ('vision' in key):
    #         print(key)

    algorithm.load_state_dict(ckpt)
    return state['epoch'], state['best_val_metric']


def load(algorithm, path):
    state = torch.load(path)
    algorithm.load_state_dict(state['algorithm'])
    return state['epoch'], state['best_val_metric']


def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')


def initialize_wandb(config):
    name = config.dataset + '_' + config.algorithm + '_' + config.log_dir
    wandb.init(name=name,
               project=f"wilds")
    wandb.config.update(config)


def save_pred(y_pred, path_prefix):
    # Single tensor
    if torch.is_tensor(y_pred):
        df = pd.DataFrame(y_pred.numpy())
        df.to_csv(path_prefix + '.csv', index=False, header=False)
    # Dictionary
    elif isinstance(y_pred, dict) or isinstance(y_pred, list):
        torch.save(y_pred, path_prefix + '.pth')
    else:
        raise TypeError("Invalid type for save_pred")


def get_replicate_str(dataset, config):
    if dataset['dataset'].dataset_name == 'poverty':
        replicate_str = f"fold:{config.dataset_kwargs['fold']}"
    else:
        replicate_str = f"seed:{config.seed}"
    return replicate_str


def get_pred_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    split = dataset['split']
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix


def get_model_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_{replicate_str}_")
    return prefix


def move_to(obj, device):
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        # Assume obj is a Tensor or other type
        # (like Batch, for MolPCBA) that supports .to(device)
        return obj.to(device)


def detach_and_clone(obj):
    if torch.is_tensor(obj):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: detach_and_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_and_clone(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        raise TypeError("Invalid type for detach_and_clone")


def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")


def remove_key(key):
    """
    Returns a function that strips out a key from a dict.
    """
    def remove(d):
        if not isinstance(d, dict):
            raise TypeError("remove_key must take in a dict")
        return {k: v for (k,v) in d.items() if k != key}
    return remove
