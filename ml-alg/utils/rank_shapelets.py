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

if __name__ == "__main__":
    # Example usage
    dataset = 'ECG5000'
    version = '_shap'
    shapelet_path = os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet_with_dtw.json')
    
    with open(shapelet_path, 'r') as f:
        shapelet_data = json.load(f)

    sorted_shapelets = sorted(
        shapelet_data,
        key=lambda x: (-x.get('imp', 0), -x.get('gain', 0))
    )
    for i, sh in enumerate(sorted_shapelets):
        sh['rank'] = i
    with open(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet_with_dtw_sorted.json'), 'w') as f:
        json.dump(shapelet_data, f, indent=2)