import os
import numpy as np
import pandas as pd
import json
from utils import get_data_file


def normalize(dataset):
    with open(get_data_file(f'./{dataset}/output_shapelet.json')) as f:
        shape_info = json.load(f)
    shape_transform = pd.read_csv(get_data_file(f'{dataset}/shapelet_transform.csv'))
    shape_norm = shape_transform.values.copy()
    for i, shape in enumerate(shape_info):
        wave_length = len(shape['wave'][0])
        print(wave_length)
        shape_norm[:, i] = shape_transform.values[:, i] / wave_length
    shape_norm_df = pd.DataFrame(shape_norm, columns=shape_transform.columns)
    shape_norm_df.to_csv(f'data/{dataset}/shapelet_transform_norm.csv', index=False)
        
if __name__ == "__main__":
    dataset = 'preterm'  # Replace with your dataset name
    normalize(dataset)