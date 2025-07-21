import os
from . import config
# import config


def get_data_file(path: str) -> str:
    target_path = os.path.join(config.DATA_ROOT, path)
    abs_path = os.path.abspath(target_path)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f'File not found: {abs_path}')
    return abs_path