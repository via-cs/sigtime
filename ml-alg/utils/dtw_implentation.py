import os
import sys
import json
import numpy as np


# self made classes and functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, 'data')
print(f"Data directory: {DATA_DIR}")



import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

def compute_dtw_distance(ts_a, ts_b):
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series.
    
    Parameters:
    ts_a (np.ndarray): First time series.
    ts_b (np.ndarray): Second time series.
    
    Returns:
    float: The DTW distance between the two time series.
    """
    alignment = dtw(ts_a, ts_b, keep_internals=True)
    return alignment.distance

def compute_shape_dtw(shapelet_data):
    shape_sims = np.zeros((len(shapelet_data), len(shapelet_data)))
    for i, sh1 in enumerate(shapelet_data):
        shape_sims[i, i] = 0  # Large value for self-similarity
        wave1 = np.array(sh1['wave'], dtype=np.float32)
        for j, sh2 in enumerate(shapelet_data):
            if i < j:
                wave2 = np.array(sh2['wave'], dtype=np.float32)
                dist = compute_dtw_distance(wave1, wave2)
                shape_sims[i, j] = dist
                shape_sims[j, i] = dist
    
    print(shape_sims)
    return shape_sims

def compute_and_save_shape_dtw(dataset, version=''):
    shapelet_path = os.path.join(DATA_DIR, f'{dataset}{version}/oshapelet_with_importance.json')
    if not os.path.exists(shapelet_path):
        shapelet_path = os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet.json')
    with open(shapelet_path, 'r') as f:
        shapelet_data = json.load(f)
    
    shape_sims = compute_shape_dtw(shapelet_data)
    for i in range(len(shapelet_data)):
        shapelet_data[i]['sims'] = shape_sims[i].tolist()
    with open(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet_with_dtw.json'), 'w') as f:
        json.dump(shapelet_data, f, indent=2)
        
if __name__ == "__main__":
   compute_and_save_shape_dtw('ECG5000', version='_shap')