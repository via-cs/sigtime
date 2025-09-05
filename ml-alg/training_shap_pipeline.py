import os
import sys
print(f"Current working directory: {os.getcwd()}")
from shap_exp.joint_pipeline import pipeline
from shap_exp.test_FCN import pipeline as shap_fcn_pipeline
from utils.dtw_implentation import compute_and_save_shape_dtw
from utils.rank_shapelets import rank_shapelets
import argparse
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
parser.add_argument('--num_shapelets_make', type=int, default=100, help='Number of shapelets to make during initialization')
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
parser.add_argument('--shap_version', type=str, default='', help='Window size for sliding window features (0 for auto)')
# --------------------------------------------------------------------------------------
# configuration loading
args = parser.parse_args()
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


# -------------------------------- Run Pipeline ---------------------------------------
# pipeline(
#     config, datatype=args.datatype, 
#     dataset=args.dataset, version=args.version,
#     training=True
# )

# -------------------------------- Shap training and evaluation ---------------------------------------

shap_fcn_pipeline(
    config, datatype=args.datatype, 
    dataset=args.dataset, version=args.version,
    shap_version=args.shap_version,
    pretrained_shapelets=True,
    training=True
)

shap_fcn_pipeline(
    config, datatype=args.datatype, 
    dataset=args.dataset, version=args.version,
    pretrained_shapelets=True,
    shap_version=args.shap_version,
    training=False
)

# -------------------------------- compute dtw ---------------------------------------
compute_and_save_shape_dtw(args.dataset, version=args.shap_version)

# -------------------------------- Ranking ---------------------------------------
rank_shapelets(args.dataset, version=args.shap_version)