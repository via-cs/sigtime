import os
import sys
import json
from collections import defaultdict
import numpy as np


# self made classes and functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(os.path.join(BASE_DIR, 'shap'), 'data')



with open(os.path.join(DATA_DIR, f'ECG200_shap/output_shapelet.json'), 'r') as f:
    shapelet_data = json.load(f)
# Add original index to preserve ordering
for i, sh in enumerate(shapelet_data):
    sh['original_index'] = i
# Group shapelets by length
shapelets_by_length = defaultdict(list)
index_mapping_model_to_json = []  # model index → original json index

for shapelet in shapelet_data:
    length = shapelet['len']
    wave = np.array(shapelet['wave'], dtype=np.float32)
    shapelets_by_length[length].append((wave, shapelet['original_index']))

# Sort lengths for model block order
sorted_lengths = sorted(shapelets_by_length.keys())
print(sorted_lengths)
print(shapelets_by_length)