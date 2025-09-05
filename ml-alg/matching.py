import torch
import numpy as np
import pandas as pd

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

def match_shapelet_to_ts(data, output_shapelet):
    """
    Match shapelets to time series data and return the minimum distances and positions.
    X_all: numpy array of shape (num_samples, seq_len)
    output_shapelet: list of dicts with keys 'wave' (numpy array) and 'len' (int)
    Returns:
        min_distance: list of numpy arrays of shape (num_samples, 1)
        match_position: list of numpy arrays of shape (1, num_samples, 2)
    """
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    
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
        
    min_distance_df = pd.DataFrame(min_distance[:, :10])
    min_distance_df.to_csv(os.path.join(output_dir, "shapelet_transform.csv"), index=False)