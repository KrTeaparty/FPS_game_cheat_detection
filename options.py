from random import seed
import os

def init_args():
    args = {
        'output_path': 'outputs/',
        'root_dir': 'outputs/',
        'log_path': 'logs/',
        'modal': 'rgb',
        'model_path': 'models/',
        'lr': '[0.0001]*3000',
        'batch_size': 64,
        'num_workers': 0,
        'num_segments': 32,
        'seed': 2025,
        'model_file': 'trans_{}.pkl'.format(2024),
        'debug': 'store_true'
    }
    return args