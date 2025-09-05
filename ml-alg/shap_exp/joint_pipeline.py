import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
import tsfel

import shap

# self made classes and functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Path to data directory
from src.fe_shape_joint import JointTraining, feature_extraction_selection, extraction_pipeline
from src.joint_model import JointModel, FeatureToOutput
from preprocessing.preterm_preprocessing import preterm_pipeline
from preprocessing.public_preprocessing import public_pipeline
from pipeline import shapelet_initialization, store_data
from utils.evaluation_and_save import eval_results

parser = argparse.ArgumentParser()
# ------------------------------ Input and Output -------------------------------------
parser.add_argument('--datatype', type=str, default='public', choices={'public', 'private'})
parser.add_argument('--dataset', type=str, default='ECG5000')
parser.add_argument('--version', type=str, default='3')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio for data splitting')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio for data splitting')
parser.add_argument('--norm_std', type=str, default='standard', choices={'standard', 'minmax'}, help='Normalization standard')
parser.add_argument('--norm_mode', type=str, default='local_before', choices={'local_before', 'global_after'}, help='Normalization mode')
parser.add_argument('--seq_min', type=int, default=15, help='Minimum sequence length')
parser.add_argument('--pad_min', type=int, default=3, help='Minimum padding length')
parser.add_argument('--step_min', type=int, default=1, help='Minimum step size')
parser.add_argument('--init_mode', type=str, default='pips', choices={'pips', 'random'}, help='Initialization mode')
parser.add_argument('--model_mode', type=str, default='JOINT', choices={'JOINT', 'LS_FCN', 'LS_Transformer', 'BOSS'}, help='Model mode')
parser.add_argument('--ws_rate', type=float, default=0.1, help='Window size rate for initialization')
parser.add_argument('--num_pip', type=float, default=0.4, help='Number of PIPs for initialization')
parser.add_argument('--num_shapelets_make', type=int, default=200, help='Number of shapelets to make during initialization')
parser.add_argument('--num_shapelets', type=int, default=50, help='Number of shapelets to use in the model')
parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--model_path', type=str, default='./model/best_model.pth', help='Path to save the best model')
parser.add_argument('--step', type=int, default=1, help='Step size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--epsilon', type=float, default=1e-7, help='Epsilon for numerical stability in the optimizer')
parser.add_argument('--k', type=int, default=6, help='Parameter k for the model')
parser.add_argument('--l1', type=float, default=1e-5, help='L1 regularization coefficient')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization coefficient')
parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file")
parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads for the model')
parser.add_argument('--d_model', type=int, default=4, help='Dimension of the model for the transformer')
parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the transformer model')



def train(
    data,
    shapelets_size_and_len,
    list_shapelets,
    list_shapelets_meta,
    dataset='ECG200',
    config={},
    version: str = '',
):
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
        
    window_size = max(shapelets_size_and_len.keys())
    window_step = config['step']
    print(window_size)
    print(X_train.shape)
    # Load or compute sliding window features
    print(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy'))
    if os.path.exists(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy')):
        X_train_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'))
        X_val_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy'))
        X_test_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy'))
    else:
        x_train = X_train.transpose(0, 2, 1)
        x_val = X_val.transpose(0, 2, 1)
        x_test = X_test.transpose(0, 2, 1)
        num_train, len_ts, in_channels = x_train.shape
        num_val = x_val.shape[0]
        num_test = x_test.shape[0]
        x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        num_windows = x_test_split.shape[1]
        x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
        x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
        x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
        cfg_file = tsfel.get_features_by_domain()
        if dataset == 'preterm':
            clean_selected_features = ['Average power', 'ECDF Percentile Count', 
                'LPCC', 'LPCC', 'LPCC', 'LPCC', 'LPCC', 
                'Median', 'Root mean square', 'Spectral distance', 
                'Spectral roll-off', 'Spectral skewness', 'Spectral slope', \
                'Spectral spread', 'Standard deviation', 'Sum absolute diff', 
                'Wavelet absolute mean_25.0Hz', 'Wavelet absolute mean_3.12Hz', 
                'Wavelet absolute mean_3.57Hz', 'Wavelet absolute mean_4.17Hz', 
                'Wavelet absolute mean_5.0Hz', 'Wavelet absolute mean_6.25Hz', 
                'Wavelet absolute mean_8.33Hz', 'Wavelet energy_25.0Hz', 'Wavelet energy_3.12Hz', 
                'Wavelet energy_3.57Hz', 'Wavelet energy_4.17Hz', 'Wavelet energy_5.0Hz', 
                'Wavelet energy_6.25Hz', 'Wavelet energy_8.33Hz', 'Wavelet standard deviation_12.5Hz', 
                'Wavelet standard deviation_2.78Hz', 'Wavelet standard deviation_25.0Hz', 'Wavelet standard deviation_3.12Hz', 
                'Wavelet standard deviation_3.57Hz', 'Wavelet standard deviation_4.17Hz', 'Wavelet standard deviation_5.0Hz', 
                'Wavelet standard deviation_6.25Hz', 'Wavelet standard deviation_8.33Hz', 'Wavelet variance_2.78Hz', 
                'Wavelet variance_3.12Hz', 'Wavelet variance_3.57Hz', 'Wavelet variance_4.17Hz', 'Wavelet variance_5.0Hz', 
                'Wavelet variance_6.25Hz', 'Wavelet variance_8.33Hz'
            ]
            # Disable all features in cfg_file first
            for domain in cfg_file.keys():
                for feature in cfg_file[domain]:
                    cfg_file[domain][feature]["use"] = "no"  # Ensure correct format

            # Enable only the selected features
            for domain in cfg_file.keys():
                for feature in cfg_file[domain]:
                    if feature in clean_selected_features:
                        cfg_file[domain][feature]["use"] =  "yes"
            X_train_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_train_split)
            X_val_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_val_split)
            X_test_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_test_split)
            scaler = StandardScaler()
            X_train_split_filtered = scaler.fit_transform(X_train_split_filtered.values)
            X_val_split_filtered = scaler.transform(X_val_split_filtered.values)
            X_test_split_filtered = scaler.transform(X_test_split_filtered.values)
            X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
            X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
            X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
        else:
            
            X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
            X_val_split = tsfel.time_series_features_extractor(cfg_file, x_val_split)
            X_test_split = tsfel.time_series_features_extractor(cfg_file, x_test_split)
            X_train_split_filtered, corr_features, selector, scaler = feature_extraction_selection(X_train_split)
            X_val_split_filtered = extraction_pipeline(X_val_split, corr_features, selector, scaler)
            X_test_split_filtered = extraction_pipeline(X_test_split, corr_features, selector, scaler)
            X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
            X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
            X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
            
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'), X_train_split_filtered)
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy'), X_val_split_filtered)
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy'), X_test_split_filtered)
    
    num_features = X_train_split_filtered.shape[-1]
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    loss_func = nn.CrossEntropyLoss()
    model = JointTraining(
        shapelets_size_and_len=shapelets_size_and_len,
        seq_len=len_ts, 
        in_channels=n_channels, 
        loss_func = loss_func, 
        mode = config['joint_mode'], 
        num_features=num_features, 
        window_size=window_size, 
        step=config['step'],
        nhead=config['nhead'], 
        num_layers=config['num_layers'],
        num_classes=num_classes, 
        to_cuda = True
    )
    print(list_shapelets)
    for i, key in enumerate(list_shapelets.keys() if list_shapelets is not None else [0, 0]):
        weights_block = []
        for j in list_shapelets[key] if list_shapelets is not None else [0]:
            weights_block.append(X_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
        weights_block = np.array(weights_block)
        model.set_shapelet_weights_of_block(i, weights_block)
    
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
    loss = model.fit(
            X_train, X_train_split_filtered, y_train,
            X_val=data['X_val'], FE_val = X_val_split_filtered, Y_val=y_val,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            shuffle=True, 
            model_path= 'model.pth'
    )
    store_data(
        data=data, 
        dataset=dataset, 
        model=model,
        list_shapelets_meta=list_shapelets_meta, 
        list_shapelets=list_shapelets, 
        output_version="shap"
    )
    torch.save(
        model.model.state_dict(), 
        os.path.join(DATA_DIR, f'model_joint_{dataset}.pth')
    )
    
    y_hat = model.predict(X_test, FE = X_test_split_filtered)
    results = eval_results(y_test, y_hat)
    
    return results

def shap_eval(
    data,
    shapelets_size_and_len,
    list_shapelets,
    list_shapelets_meta,
    dataset='ECG200',
    config={},
    version: str = '',
):
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
        
    window_size = max(shapelets_size_and_len.keys())
    window_step = config['step']
    
    # Load or compute sliding window features
    if os.path.exists(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy')):
        X_train_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'))
        X_val_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy'))
        X_test_split_filtered = np.load(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy'))
    else:
        x_train = X_train.transpose(0, 2, 1)
        x_val = X_val.transpose(0, 2, 1)
        x_test = X_test.transpose(0, 2, 1)
        num_train, len_ts, in_channels = x_train.shape
        num_val = x_val.shape[0]
        num_test = x_test.shape[0]
        x_train_split = sliding_window_view(x_train, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        x_val_split = sliding_window_view(x_val, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        x_test_split = sliding_window_view(x_test, window_size, axis=1).transpose(0, 1, 3, 2)[:, ::window_step]
        num_windows = x_test_split.shape[1]
        x_train_split = x_train_split.reshape(num_train * num_windows, window_size, in_channels)
        x_val_split = x_val_split.reshape(num_val * num_windows, window_size, in_channels)
        x_test_split = x_test_split.reshape(num_test * num_windows, window_size, in_channels)
        cfg_file = tsfel.get_features_by_domain()
        if dataset == 'preterm':
            clean_selected_features = ['Average power', 'ECDF Percentile Count', 
                'LPCC', 'LPCC', 'LPCC', 'LPCC', 'LPCC', 
                'Median', 'Root mean square', 'Spectral distance', 
                'Spectral roll-off', 'Spectral skewness', 'Spectral slope', \
                'Spectral spread', 'Standard deviation', 'Sum absolute diff', 
                'Wavelet absolute mean_25.0Hz', 'Wavelet absolute mean_3.12Hz', 
                'Wavelet absolute mean_3.57Hz', 'Wavelet absolute mean_4.17Hz', 
                'Wavelet absolute mean_5.0Hz', 'Wavelet absolute mean_6.25Hz', 
                'Wavelet absolute mean_8.33Hz', 'Wavelet energy_25.0Hz', 'Wavelet energy_3.12Hz', 
                'Wavelet energy_3.57Hz', 'Wavelet energy_4.17Hz', 'Wavelet energy_5.0Hz', 
                'Wavelet energy_6.25Hz', 'Wavelet energy_8.33Hz', 'Wavelet standard deviation_12.5Hz', 
                'Wavelet standard deviation_2.78Hz', 'Wavelet standard deviation_25.0Hz', 'Wavelet standard deviation_3.12Hz', 
                'Wavelet standard deviation_3.57Hz', 'Wavelet standard deviation_4.17Hz', 'Wavelet standard deviation_5.0Hz', 
                'Wavelet standard deviation_6.25Hz', 'Wavelet standard deviation_8.33Hz', 'Wavelet variance_2.78Hz', 
                'Wavelet variance_3.12Hz', 'Wavelet variance_3.57Hz', 'Wavelet variance_4.17Hz', 'Wavelet variance_5.0Hz', 
                'Wavelet variance_6.25Hz', 'Wavelet variance_8.33Hz'
            ]
            # Disable all features in cfg_file first
            for domain in cfg_file.keys():
                for feature in cfg_file[domain]:
                    cfg_file[domain][feature]["use"] = "no"  # Ensure correct format

            # Enable only the selected features
            for domain in cfg_file.keys():
                for feature in cfg_file[domain]:
                    if feature in clean_selected_features:
                        cfg_file[domain][feature]["use"] =  "yes"
            X_train_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_train_split)
            X_val_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_val_split)
            X_test_split_filtered = tsfel.time_series_features_extractor(cfg_file, x_test_split)
            scaler = StandardScaler()
            X_train_split_filtered = scaler.fit_transform(X_train_split_filtered.values)
            X_val_split_filtered = scaler.transform(X_val_split_filtered.values)
            X_test_split_filtered = scaler.transform(X_test_split_filtered.values)
            X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
            X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
            X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
        else:
            
            X_train_split = tsfel.time_series_features_extractor(cfg_file, x_train_split)
            X_val_split = tsfel.time_series_features_extractor(cfg_file, x_val_split)
            X_test_split = tsfel.time_series_features_extractor(cfg_file, x_test_split)
            X_train_split_filtered, corr_features, selector, scaler = feature_extraction_selection(X_train_split)
            X_val_split_filtered = extraction_pipeline(X_val_split, corr_features, selector, scaler)
            X_test_split_filtered = extraction_pipeline(X_test_split, corr_features, selector, scaler)
            X_train_split_filtered = X_train_split_filtered.reshape(num_train, num_windows, -1)
            X_val_split_filtered = X_val_split_filtered.reshape(num_val, num_windows, -1)
            X_test_split_filtered = X_test_split_filtered.reshape(num_test, num_windows, -1)
            
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'), X_train_split_filtered)
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy'), X_val_split_filtered)
        np.save(os.path.join(DATA_DIR, f'{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy'), X_test_split_filtered)
    
    num_features = X_train_split_filtered.shape[-1]
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    model = JointModel(
        shapelets_size_and_len=shapelets_size_and_len,
        seq_len=len_ts,
        num_features=num_features,
        mode=config['joint_mode'],
        window_size=window_size,
        in_channels=n_channels,
        step=config['step'],
        num_classes=num_classes,
        dist_measure='euclidean',
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        batch_first=True,
        to_cuda=False
    )
    for i, key in enumerate(list_shapelets.keys() if list_shapelets is not None else [0, 0]):
        weights_block = []
        for j in list_shapelets[key] if list_shapelets is not None else [0]:
            weights_block.append(X_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
        weights_block = np.array(weights_block, dtype=np.float32)
        model.shapelets_blocks.set_shapelet_weights_of_block(i, weights_block)
    
    model.load_state_dict(
        torch.load(os.path.join(DATA_DIR, f'model_joint_{dataset}.pth'), weights_only=True)
    )
    model.eval()
    feature2out = FeatureToOutput(
        joint_feature=model.joint_feature,                     # shares weights
        positional_emb=model.positional_emb,                   # shares weights
        transformer_encoder=model.transformer_encoder,         # shares weights
        linear=model.linear,                                   # shares weights
        transform_seq_len=model.transform_seq_len,
        mode=model.mode
    )
    feature2out.eval()
    
    
    N_bg = 10
    seq_len = X_train_split_filtered.shape[-2]            # as used in your model
    num_shapelets = sum(shapelets_size_and_len.values())
    num_features = X_train_split_filtered.shape[-1]
    
    y_bg = torch.zeros((N_bg, seq_len, num_shapelets), dtype=torch.float32)
    stat_bg = torch.zeros((N_bg, seq_len, num_features), dtype=torch.float32)  
    
    N_test = 10
    _, y_test = model.shapelets_blocks(torch.tensor(X_train[:N_test], dtype=torch.float32))
    y_test = torch.squeeze(y_bg, 1)
    # Use the same normalization as in your forward
    stat_test = torch.tensor(X_train_split_filtered[:N_test], dtype=torch.float32)
    
    y_bg_flat = y_bg.cpu().numpy().reshape(N_bg, -1)
    stat_bg_flat = stat_bg.cpu().numpy().reshape(N_bg, -1)
    background = np.concatenate([y_bg_flat, stat_bg_flat], axis=1)  # shape: [N_bg, total_features]

    # For your test set (should be the same as above!)
    # y_test: [N_test, seq_len, num_shapelets]
    # stat_test: [N_test, seq_len, num_features]
    y_test_flat = y_test.cpu().numpy().reshape(N_test, -1)
    stat_test_flat = X_train_split_filtered[:N_test].reshape(N_test, -1)  # if it's numpy, otherwise .cpu().numpy() if tensor
    test = np.concatenate([y_test_flat, stat_test_flat], axis=1)
    def predict_fn(X):
        # X: [batch, total_features] numpy array
        # You need to split it back into y and stat, and reshape

        # Figure out the sizes
        n = X.shape[0]
        y_size = seq_len * num_shapelets
        stat_size = seq_len * num_features

        y_flat = X[:, :y_size]
        stat_flat = X[:, y_size:y_size+stat_size]
        y = torch.tensor(y_flat, dtype=torch.float32).reshape(n, seq_len, num_shapelets).to(device)
        stat = torch.tensor(stat_flat, dtype=torch.float32).reshape(n, seq_len, num_features).to(device)
        with torch.no_grad():
            out = feature2out(y, stat)
        return out.cpu().numpy()

    device = next(feature2out.parameters()).device

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test)
    
    print(np.shape(shap_values[0]))  
    print(np.shape(shap_values[1])) 
    print(len(shap_values))
    
    np.save(os.path.join(DATA_DIR, f'shap_values_0_{dataset}.npy'), shap_values[0])
    np.save(os.path.join(DATA_DIR, f'shap_values_1_{dataset}.npy'), shap_values[1])
    

def pipeline(config, dataset='ECG200', datatype='public', version='', training = True):
    
    data_path = os.path.join(DATA_DIR, f'{dataset}.npz')
    print(data_path)
    meta_path = os.path.join(DATA_DIR, 'filtered_clinical_data.csv')
    strip_path='./data/filtered_strips_data.json'
    if len(version) > 0 and datatype == 'private':
        data_path = os.path.join('./data', f'{dataset}_v{version}.npz')
        meta_path = os.path.join(DATA_DIR, f'filtered_clinical_data_v{version}.csv')
        strip_path = os.path.join(DATA_DIR, f'filtered_strips_data_v{version}.json')
    
    if datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            meta_path=meta_path, 
            strip_path=strip_path,
            data_path=data_path
        )
    
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=False, 
            root=DATA_DIR,
            config=config['data_loading'], 
        )
        
    X_train = data['X_train']
    y_train = data['y_train']
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    shapelets_size_and_len, list_shapelets_meta, list_shapelets, shapetime = \
        shapelet_initialization(
            X_train, y_train, 
            config=config['init_config'], 
            dataset=dataset, 
            mode=config['init_mode'],
            root=DATA_DIR,
            version=version
        )
    if training:
        final_results = train(
            data,
            shapelets_size_and_len=shapelets_size_and_len,
            list_shapelets=list_shapelets,
            list_shapelets_meta=list_shapelets_meta,
            config=config['model_config'],
            version=version, 
            dataset=dataset
        )
        print(final_results)
    else:
        shap_eval(
            data,
            shapelets_size_and_len=shapelets_size_and_len,
            list_shapelets=list_shapelets,
            list_shapelets_meta=list_shapelets_meta,
            config=config['model_config'],
            version=version,
            dataset=dataset
        )

if __name__ == "__main__":
    
    args = parser.parse_args()
    dataset = args.dataset
    datatype = args.datatype
    store_results = False
    
    config = { # default
        'data_loading': {
            'test_ratio': args.test_ratio,
            'val_ratio': args.val_ratio,
            'norm_std': args.norm_std,
            'norm_mode': args.norm_mode,
            'seq_min': args.seq_min,
            'pad_min': args.pad_min,
            'step_min': args.step_min,
        },
        'init_mode': args.init_mode, # 'pips' / 'random'
        'model_mode': args.model_mode, # 'JOINT' / 'LS_FCN' / 'LS_Transformer' / 'BOSS'
        'init_config': {
            'size_ratio': [0.1, 0.2], 
            'ws_rate': args.ws_rate,
            'num_pip': args.num_pip, 
            'num_shapelets_make': args.num_shapelets_make, 
            'num_shapelets': args.num_shapelets,
        },

        'model_config': {
            'joint_mode': 'concat',
            'nhead':args.nhead,
            'd_model': args.d_model,
            'num_layers': args.num_layers,
            'shuffle': True,
            'epochs': args.epochs, 
            'batch_size': args.batch_size, 
            'model_path': './model/best_model.pth',
            'step': args.step,
            'lr': args.lr, 
            'wd': args.wd, 
            'epsilon': args.epsilon,
            'k': args.k,
            'l1': args.l1,
            'l2': args.l2
        },
    }

    data_path = os.path.join(DATA_DIR, f'{dataset}.npz')
    print(args.dataset)
    if datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            data_path=data_path
        )
    
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=store_results, 
            root=DATA_DIR,
            config=config['data_loading'], 
        )
        
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    pipeline(
        config, datatype=args.datatype, 
        dataset=args.dataset, version='',
        training=False
    )
    
    # print(results)