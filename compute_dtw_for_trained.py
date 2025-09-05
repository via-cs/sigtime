import os
import sys
print(f"Current working directory: {os.getcwd()}")
from utils.dtw_implentation import compute_and_save_shape_dtw
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ECG5000')
parser.add_argument('--shap_version', type=str, default='ECG5000')
# configuration loading
args = parser.parse_args()
compute_and_save_shape_dtw(args.dataset, version=args.shap_version)