# os and environment
import os
import sys
sys.path.insert(0, os.getcwd())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, 'data')

import time
import random
import json

# data handling
import numpy as np
import pandas as pd

# machine learning libraries
import torch
from torch import nn, optim
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import tsfel
from numpy.lib.stride_tricks import sliding_window_view

# self-defined modules
from preprocessing.preterm_preprocessing import preterm_pipeline
from preprocessing.public_preprocessing import public_pipeline
from shapelet_candidate.mul_shapelet_discovery import ShapeletDiscover
from src.learning_shapelets_DTW import LearningShapelets as LearningShapeletsFCN
from src.learning_shapelets_sliding_window import LearningShapelets as LearningShapeletsTranformer
from src.fe_shape_joint_dtw import JointTraining
from src.fe_shape_joint_dtw import feature_extraction_selection, extraction_pipeline
from utils.evaluation_and_save import eval_results 

torch.cuda.set_device(0)

def torch_dist_ts_shapelet(ts, shapelet, cuda=True):
    """
    Calculate euclidean distance of shapelet to a time series via PyTorch and returns the distance along with the position in the time series.
    """
    
    if not isinstance(ts, torch.Tensor):
        ts = torch.tensor(ts, dtype=torch.float)
    if not isinstance(shapelet, torch.Tensor):
        shapelet = torch.tensor(shapelet, dtype=torch.float)
    if cuda:
        ts = ts.cuda()
        shapelet = shapelet.cuda()
    shapelet = torch.unsqueeze(shapelet, 0)
    # Remove trailing NaNs from shapelet
    if torch.isnan(shapelet).any():
        # Find the last non-NaN index along the last dimension
        valid_length = (~torch.isnan(shapelet[0, 0])).sum().item()
        shapelet = shapelet[:, :, :valid_length]
    # unfold time series to emulate sliding window\
    ts = ts.unfold(1, shapelet.shape[2], 1)
    # calculate euclidean distance
    dists = torch.cdist(ts, shapelet, p=2)
    dists = torch.sum(dists, dim=0)
    # otherwise gradient will be None
    # hard min compared to soft-min from the paper
    d_min, d_argmin = torch.min(dists, 0)
    return (d_min.item(), d_argmin.item())

def shapelet_initialization(
    X_train, 
    y_train, 
    config,
    dataset='preterm',
    mode='pips', 
    root='./data',
    version=''
):
    _, n_chaanels, len_ts = X_train.shape
    if mode == 'pips':
        ws_rate = config['ws_rate']
        num_pip = config['num_pip']
        num_shapelets_make = config['num_shapelets_make'] # number of shapelets to make for each label
        num_shapelets = config['num_shapelets']
        
        t1 = time.time()
        csv_path = os.path.join(root, f'list_shapelets_meta_{dataset}_{ws_rate}_{num_pip}_{num_shapelets_make}.csv')
        if os.path.exists(csv_path):
            
            df_shapelets_meta = pd.read_csv(csv_path)
            elapsed_time = time.time() - t1

        else:

            t1 = time.time()
            shapelet = ShapeletDiscover(
                window_size = int(len_ts * ws_rate),
                num_pip= num_pip,
            )
            shapelet.extract_candidate(X_train)
            shapelet.discovery(X_train, y_train)
            list_shapelets_meta = shapelet.get_shapelet_info(number_of_shapelet=num_shapelets_make) 
            
            if list_shapelets_meta is not None:
                list_shapelets_meta = list_shapelets_meta[list_shapelets_meta[:, 3].argsort()[::-1]]
            else:
                raise ValueError("ShapeletDiscover.get_shapelet_info returned None.")
            elapsed_time = time.time() - t1
        
            df_shapelets_meta = pd.DataFrame(
                list_shapelets_meta, columns=[
                    'series_position', 'start_pos', 'end_pos', 
                    'inforgain', 'label', 'dim']
                )
            df_shapelets_meta = df_shapelets_meta.sort_values(by='inforgain', ascending=False)
            print(df_shapelets_meta)  # Display first 5 shapelets for brevity
            # Filter out long shapelets
            max_shapelet_length = int(len_ts * 0.5)  # filter out shapelets longer than 50% of the time series length
            
            df_shapelets_meta = df_shapelets_meta[
                df_shapelets_meta['end_pos'] - df_shapelets_meta['start_pos'] <= max_shapelet_length
            ]
            df_shapelets_meta.to_csv(csv_path, index=False)
        
        if num_shapelets > len(df_shapelets_meta):
            list_shapelets_meta = df_shapelets_meta.values
        else:
            list_shapelets_meta = df_shapelets_meta.values[:num_shapelets]
            
        list_shapelets = {}
        for i in range(list_shapelets_meta.shape[0] if list_shapelets_meta is not None else 0):
            shape_size = int(list_shapelets_meta[i, 2] - int(list_shapelets_meta[i, 1]))
            if shape_size not in list_shapelets:
                list_shapelets[shape_size] = [i]
            else:
                list_shapelets[shape_size].append(i)

        list_shapelets = {key: list_shapelets[key] for key in sorted(list_shapelets)} 
        shapelets_size_and_len = dict()
        for i in list_shapelets.keys():
            shapelets_size_and_len[i] = len(list_shapelets[i])
        return shapelets_size_and_len, list_shapelets_meta, list_shapelets, elapsed_time
    
    else:
        size_ratio = config['size_ratio']
        num_shapelets = config['num_shapelets']
        shapelets_size_and_len = dict()
        for i in size_ratio:
            size = int(len_ts * i)
            shapelets_size_and_len[size] = num_shapelets
        
        return shapelets_size_and_len
def sample_ts_segments(X, shapelets_size, n_segments=1000):
    """
    Sample time series segments for k-Means.
    """
    n_ts, n_channels, len_ts = X.shape
    samples_i = random.choices(range(n_ts), k=n_segments)
    segments = np.empty((n_segments, n_channels, shapelets_size))
    for i, k in enumerate(samples_i):
        s = random.randint(0, len_ts - shapelets_size)
        segments[i] = X[k, :, s:s+shapelets_size]
    return segments
def get_weights_via_kmeans(X, shapelets_size, num_shapelets, n_segments=1000):
    """
    Get weights via k-Means for a block of shapelets.
    """
    segments = sample_ts_segments(X, shapelets_size, n_segments).transpose(0, 2, 1)
    k_means = TimeSeriesKMeans(n_clusters=num_shapelets, metric="euclidean", max_iter=50).fit(segments)
    clusters = k_means.cluster_centers_.transpose(0, 2, 1)
    return clusters

def store_data(data, dataset, model, list_shapelets_meta, list_shapelets, output_version=''):
    
    output_dir = f"./data/{dataset}_{output_version}"
    os.makedirs(output_dir, exist_ok=True)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)
    shapelets = model.get_shapelets()
    
    i = 0
    output_shapelet = []
    for key in sorted(list_shapelets.keys()):
        for idx in list_shapelets[key]:
            shape_len = int(key)
            wave = shapelets[i, :, :shape_len]
            shape_info = {
                'len': shape_len,
                'gain': list_shapelets_meta[idx, 3],
                'wave': shapelets[i, :, :shape_len]
            }
            i += 1
            output_shapelet.append(shape_info)
    
    match_position = []
    min_distance = []
    for i in range(len(output_shapelet)):
        d_min = []
        pos_start = []
        for j in range(X_all.shape[0]):
            d_min_j, pos_start_j = torch_dist_ts_shapelet(X_all[j], output_shapelet[i]['wave'])
            d_min.append(d_min_j)
            pos_start.append(pos_start_j)
        pos_start = np.array(pos_start).reshape(len(pos_start), 1)
        pos_end = np.zeros(pos_start.shape)
        pos_end = pos_start + output_shapelet[i]['len']
        pos = np.concatenate((pos_start, pos_end), axis=1)
        pos = np.expand_dims(pos, 0)
        d_min = np.array(d_min).reshape(len(d_min), 1)
        match_position.append(pos)
        min_distance.append(d_min)
    
    match_position = np.concatenate(match_position)
    min_distance = np.concatenate(min_distance, axis=1)
    print(min_distance.shape)
    match_position = np.transpose(match_position, (1, 0, 2))
    
    
    # Sort match_position based on output_shapelet['gain'] on axis 1
    gains = np.array([shapelet['gain'] for shapelet in output_shapelet])
    sorted_indices = np.argsort(gains)[::-1]
    match_position = match_position[:, sorted_indices, :]
    min_distance = min_distance[:, sorted_indices]
    
    match_position_start = match_position[:, :, 0].reshape(match_position.shape[0], match_position.shape[1])
    match_position_end = match_position[:, :, 1].reshape(match_position.shape[0], match_position.shape[1])
    # Sort output_shapelet based on 'gain'
    output_shapelet = sorted(output_shapelet, key=lambda x: x['gain'], reverse=True)
    output_shapelet = output_shapelet[:10]  # Keep only the top 10 shapelets based on gain
    
    
    
    min_distance_df = pd.DataFrame(min_distance[:, :10])
    min_distance_df.to_csv(os.path.join(output_dir, "shapelet_transform.csv"), index=False)
    X_all_df = pd.DataFrame(X_all.reshape(X_all.shape[0], -1))
    X_all_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_all_df = pd.DataFrame(y_all, columns=['label'])
    y_all_df.to_csv(os.path.join(output_dir, "label.csv"), index=False)
    match_start_df = pd.DataFrame(match_position_start[:, :10])
    match_start_df.to_csv(os.path.join(output_dir, "match_start.csv"), index=False)
    match_end_df = pd.DataFrame(match_position_end[:, :10])
    match_end_df.to_csv(os.path.join(output_dir, "match_end.csv"), index=False)
    output_shapelet_json = [
        {
            'len': shapelet['len'],
            'gain': shapelet['gain'],
            'wave': shapelet['wave'].tolist()
        }
        for shapelet in output_shapelet
    ]

    with open(os.path.join(output_dir, "output_shapelet.json"), 'w') as f:
        json.dump(output_shapelet_json, f)
        
def train(
    data,
    shapelets_size_and_len,
    init_mode,
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
    if os.path.exists(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy'):
        X_train_split_filtered = np.load(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy')
        X_val_split_filtered = np.load(f'./data/{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy')
        X_test_split_filtered = np.load(f'./data/{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy')
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
            
        np.save(f'./data/{dataset}_X_train_split_filtered_{window_size}_{window_step}_{version}.npy', X_train_split_filtered)
        np.save(f'./data/{dataset}_X_val_split_filtered_{window_size}_{window_step}_{version}.npy', X_val_split_filtered)
        np.save(f'./data/{dataset}_X_test_split_filtered_{window_size}_{window_step}_{version}.npy', X_test_split_filtered)
    
    num_features = X_train_split_filtered.shape[-1]
    print(num_features)
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    loss_func = nn.CrossEntropyLoss()
    # model = JointTraining(
    #     shapelets_size_and_len=shapelets_size_and_len,
    #     seq_len=len_ts, 
    #     in_channels=n_channels, 
    #     loss_func = loss_func, 
    #     mode = config['joint_mode'], 
    #     num_features=num_features, 
    #     window_size=window_size, 
    #     step=config['step'],
    #     nhead=config['nhead'], 
    #     num_layers=config['num_layers'],
    #     num_classes=num_classes, 
    #     to_cuda = True
    # )
    _, n_channels, len_ts = X_train.shape
    print("batch size:", config['batch_size'])
    loss_func = nn.CrossEntropyLoss()
    model = LearningShapeletsFCN(
        shapelets_size_and_len=shapelets_size_and_len,
        loss_func=loss_func,
        in_channels=n_channels,
        num_classes=num_classes,
    )
    
    window_size = max(shapelets_size_and_len.keys())
    t1 = time.time()
    
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
    
    model_path = f'./model/{dataset}_{init_mode}_{version}.pt'
    loss =  model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            shuffle=True, 
            model_path= model_path
    )
    
    torch.save(
        model.model.state_dict(), 
        os.path.join(DATA_DIR, f'model_dtw_{dataset}.pth')
    )
    # evaluation
    y_hat = model.predict(X_test)
    results = eval_results(y_test, y_hat)
    elapsed_time = time.time() - t1
    store_data(data, dataset, model, list_shapelets_meta, list_shapelets)
    
    store_data(data, dataset, model, list_shapelets_meta, list_shapelets, output_version='dtw')
    
    
    return elapsed_time, results, loss[-1]


def pipeline(config, dataset='ECG200', datatype='public', version=''):
    store_results = False
    
    data_path = os.path.join('./data', f'{dataset}.npz')
    meta_path='./data/filtered_clinical_data.csv'
    strip_path='./data/filtered_strips_data.json'
    if len(version) > 0 and datatype == 'private':
        data_path = os.path.join('./data', f'{dataset}_v{version}.npz')
        meta_path=f'./data/filtered_clinical_data_v{version}.csv'
        strip_path=f'./data/filtered_strips_data_v{version}.json' 
    
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
            output=store_results, 
            root='./data',
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
            mode=config['init_mode']
        )
    final_results = train(
            data,
            shapelets_size_and_len=shapelets_size_and_len,
            list_shapelets=list_shapelets,
            list_shapelets_meta=list_shapelets_meta,
            init_mode='pips',
            dataset=dataset,
            config=config['model_config'],
            version=version
        )
    elapsed = final_results[0]
    results = final_results[1]
    val_loss = final_results[2]
    print(f"Shapelet initialization took {shapetime:.2f} seconds.")
    print(f"Shapelets size and length: {shapelets_size_and_len}")
    print(f"Number of shapelets: {len(list_shapelets_meta)}")
    print(list_shapelets_meta)  # Disp
    
    return elapsed, results, val_loss, shapetime
if __name__ == "__main__":
    
    dataset = 'preterm'
    datatype = 'private'
    store_results = False
    
    config = {
        'data_loading': {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
            'norm_mode': 'local_before',
            'seq_min': 15,
            'pad_min': 3, 
            'step_min': 1,
        },
        'init_mode': 'pips',
        'init_config': {
            'ws_rate': 0.1,
            'num_pip': 0.4, 
            'num_shapelets_make': 100, 
            'num_shapelets': 20,
        },
    }
    data_path = os.path.join('./data', f'{dataset}.npz')
    
    if datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            data_path=data_path
        )
    
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=store_results, 
            root='./data',
            config=config['data_loading'], 
        )
        
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    shapelets_size_and_len, list_shapelets_meta, list_shapelets, shapetime = \
        shapelet_initialization(
            X_train, y_train, 
            config=config['init_config'], 
            dataset=dataset, 
            mode=config['init_mode']
        )

    print(f"Shapelet initialization took {shapetime:.2f} seconds.")
    print(f"Shapelets size and length: {shapelets_size_and_len}")
    print(f"Number of shapelets: {len(list_shapelets_meta)}")
    print(list_shapelets_meta)  # Display first 5 shapelets for brevity