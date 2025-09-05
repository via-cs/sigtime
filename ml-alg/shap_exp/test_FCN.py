import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import optim
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

import shap

# self made classes and functions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Path to data directory
from src.fe_shape_joint import JointTraining, feature_extraction_selection, extraction_pipeline
from src.joint_model import JointModel, FeatureToOutput
from src.learning_shapelets import LearningShapelets, LearningShapeletsModel
from src.learning_shapelets_shap import ShapeletSHAPWrapper
from preprocessing.preterm_preprocessing import preterm_pipeline
from preprocessing.public_preprocessing import public_pipeline
from pipeline import shapelet_initialization, store_data
from utils.evaluation_and_save import eval_results

parser = argparse.ArgumentParser()
# ------------------------------ Input and Output -------------------------------------
parser.add_argument('--datatype', type=str, default='public', choices={'public', 'private'})
parser.add_argument('--dataset', type=str, default='ECG5000')
parser.add_argument('--version', type=str, default='')
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
parser.add_argument('--training', action='store_true', help='Flag to indicate training mode')
parser.add_argument('--pretrain', action='store_true', help='Flag to indicate whether to use pretrained shapelets')
def train(
    data,
    shapelets_size_and_len,
    list_shapelets,
    list_shapelets_meta,
    dataset='ECG200',
    config={},
    pretrained_shapelets=True,
    version: str = '_shap',
):
    print(pretrained_shapelets)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
        
    window_size = max(shapelets_size_and_len.keys())
    window_step = config['step']
    print(dataset)
    # Load or compute sliding window features
    
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    loss_func = nn.CrossEntropyLoss()
    print(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet.json'))
    if pretrained_shapelets:
        with open(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet.json'), 'r') as f:
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
        # Feed into model + build index mapping
        shapelets_size_and_len = {
            length: len(shapelets_by_length[length])
            for length in sorted(shapelets_by_length)
        }
    model = LearningShapelets(
        loss_func=loss_func,
        shapelets_size_and_len=shapelets_size_and_len,
        in_channels=n_channels,
        num_classes=num_classes,
        k=config['k'],
        l1=config['l1'],
        l2=config['l2'],
        to_cuda=True
    )
    
    if pretrained_shapelets:
        for block_idx, length in enumerate(sorted_lengths):
            shapelets_in_block = shapelets_by_length[length]
            block_waves = []
            for wave, orig_idx in shapelets_in_block:
                block_waves.append(wave)  # add channel dim if needed
                index_mapping_model_to_json.append(orig_idx)

            block_waves = np.stack(block_waves) # [num_shapelets, 1, length]
            model.set_shapelet_weights_of_block(block_idx, block_waves)
    else:
        for i, key in enumerate(list_shapelets.keys() if list_shapelets is not None else [0, 0]):
            weights_block = []
            for j in list_shapelets[key] if list_shapelets is not None else [0]:
                weights_block.append(X_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
            weights_block = np.array(weights_block)
            model.set_shapelet_weights_of_block(i, weights_block)
    
    optimizer = optim.Adam(model.model.parameters(), lr=config['lr'], weight_decay=config['wd'], eps=config['epsilon'])
    model.set_optimizer(optimizer)
    loss = model.fit(
        X_train, y_train, X_val=X_val, Y_val=y_val, shuffle=config['shuffle'],
        epochs=config['epochs'], batch_size=config['batch_size'],
        model_path='model.pth'
    )
    if not pretrained_shapelets:
        store_data(
            data=data, 
            dataset=dataset, 
            model=model,
            list_shapelets_meta=list_shapelets_meta, 
            list_shapelets=list_shapelets, 
            output_version="shap"
        )
    print(shapelets_size_and_len)
    print(os.path.join(DATA_DIR, f'model_FCN_{dataset}.pth'))
    torch.save(
        model.model.state_dict(), 
        os.path.join(DATA_DIR, f'model_FCN_{dataset}.pth')
    )
    
    y_hat = model.predict(X_test)
    results = eval_results(y_test, y_hat)
    
    return results

def shap_eval(
    data,
    shapelets_size_and_len,
    list_shapelets,
    list_shapelets_meta,
    dataset='ECG200',
    config={},
    pretrained_shapelets=True,
    version: str = '',
):
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(pretrained_shapelets)
    print(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet.json'))
    _, n_channels, len_ts = X_train.shape
    num_classes = len(set(y_train))
    if pretrained_shapelets:
        with open(os.path.join(DATA_DIR, f'{dataset}{version}/output_shapelet.json'), 'r') as f:
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
        # Feed into model + build index mapping
        shapelets_size_and_len = {
            length: len(shapelets_by_length[length])
            for length in sorted(shapelets_by_length)
        }
    model = LearningShapeletsModel(
        shapelets_size_and_len, 
        in_channels=n_channels,
        num_classes=num_classes, 
        dist_measure='euclidean',
        to_cuda=False
    )
    if pretrained_shapelets:
        for block_idx, length in enumerate(sorted_lengths):
            shapelets_in_block = shapelets_by_length[length]
            block_waves = []
            for wave, orig_idx in shapelets_in_block:
                block_waves.append(wave)  # add channel dim if needed
                index_mapping_model_to_json.append(orig_idx)

            block_waves = np.stack(block_waves) # [num_shapelets, 1, length]
            model.set_shapelet_weights_of_block(block_idx, block_waves)
    else:
        for i, key in enumerate(list_shapelets.keys() if list_shapelets is not None else [0, 0]):
            weights_block = []
            for j in list_shapelets[key] if list_shapelets is not None else [0]:
                weights_block.append(X_train[int(list_shapelets_meta[j, 0]), :, int(list_shapelets_meta[j, 1]):int(list_shapelets_meta[j, 2])])
            weights_block = np.array(weights_block)
            model.set_shapelet_weights_of_block(i, weights_block)
    print(shapelets_size_and_len)
    print(os.path.join(DATA_DIR, f'model_FCN_{dataset}.pth'))
    model.load_state_dict(
        torch.load(os.path.join(DATA_DIR, f'model_FCN_{dataset}.pth'), weights_only=True)
    )
    
    # Original model
    model.cpu()
    model.eval()

    # Step 1: Get shapelet features\
    input_tensor = torch.tensor(X_train, dtype=torch.float32).to(next(model.parameters()).device)
    input_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(next(model.parameters()).device)
    with torch.no_grad():
        shapelet_features = model.shapelets_blocks(input_tensor)  # shape: [N, 1, num_shapelets]
        shapelet_features = torch.squeeze(shapelet_features, 1)  # [N, num_shapelets]
        shapelet_features_test = model.shapelets_blocks(input_tensor_test)  # shape: [N, 1, num_shapelets]
        shapelet_features_test = torch.squeeze(shapelet_features_test, 1)  

    # Step 2: Wrap the linear layer
    wrapped_model = ShapeletSHAPWrapper(model.linear)
    wrapped_model.eval()

    # Step 3: Use SHAP
    def predict_fn(input_numpy):  # input: [batch_size, num_shapelets]
        input_tensor = torch.tensor(input_numpy, dtype=torch.float32)
        with torch.no_grad():
            out = wrapped_model(input_tensor)
            probs = torch.softmax(out, dim=1)
        return probs.numpy()

    # background: [N_bg, num_shapelets]
    background = shapelet_features.cpu().numpy()  # background for KernelExplainer
    test = shapelet_features_test.cpu().numpy()     # test set

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test)
    print(shap_values.shape)
    global_importance = np.mean(
        np.abs(np.array(shap_values)), axis=(0, 2)
    )  # shape: (num_shapelets,)
    print("Global feature importance (shapelet level):", global_importance)
    if pretrained_shapelets:
        for model_idx, json_idx in enumerate(index_mapping_model_to_json):
            shapelet_data[json_idx]['imp'] = float(global_importance[model_idx])
        # Sort if you want to keep the original gain order
        shapelet_data_sorted = sorted(shapelet_data, key=lambda s: -s['gain'])

        # Save
        with open(os.path.join(DATA_DIR,f'{dataset}{version}/shapelet_with_importance.json'), 'w') as f:
            json.dump(shapelet_data_sorted, f, indent=2)


def pipeline(config, dataset='ECG200', datatype='public', version='', training = True, pretrained_shapelets=True, shap_version: str = ''):
    
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
            version=shap_version, 
            dataset=dataset,
            pretrained_shapelets=pretrained_shapelets
        )
        print(final_results)
    else:
        shap_eval(
            data,
            shapelets_size_and_len=shapelets_size_and_len,
            list_shapelets=list_shapelets,
            list_shapelets_meta=list_shapelets_meta,
            config=config['model_config'],
            version=shap_version,
            dataset=dataset,
            pretrained_shapelets=pretrained_shapelets
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
        dataset=args.dataset, shap_version=args.version,
        training=args.training, pretrained_shapelets=args.pretrain
    )