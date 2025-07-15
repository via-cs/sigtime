import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from src.fe_shape_joint import JointModel
from src.learning_shapelets_sliding_window import ShapeletsDistBlocks

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
    
class JointModel(nn.Module):
    """
    From the input sequence, out a sequence of selected features.
    """
    def __init__(
        self, 
        shapelets_size_and_len, 
        seq_len, 
        num_features, 
        mode = 'fusion',
        window_size = 30, 
        in_channels = 1, 
        step = 1, 
        num_classes = 2, 
        dist_measure = 'euclidean',
        nhead = 2, 
        num_layers = 4, 
        batch_first = True,
        to_cuda=True
    ):
        super(JointModel, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.num_features = num_features
        self.d_model = self.num_shapelets
        self.mode = mode
        print(seq_len, window_size, step)
        self.transform_seq_len = int((seq_len - window_size)/step+1)
        
        
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    step=step, window_size=window_size,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda)
        
        self.proj_1 = nn.Linear(self.num_shapelets, self.d_model)
        self.proj_2 = nn.Linear(self.num_features, self.d_model)
        self.fusion_weights = nn.Linear(self.d_model * 2, 2)
        
        self.positional_emb = nn.Embedding(self.transform_seq_len, self.d_model)
        print(self.d_model, nhead)
        encoder_layers = TransformerEncoderLayer(d_model=self.d_model, nhead = nhead, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.concat_layer = nn.Linear(self.d_model*2, self.d_model)    
        # Optional fusion network
        self.linear = nn.Linear(self.d_model, num_classes)
        self.max = 0
        if self.to_cuda:
            self.cuda()
    def forward(self, x, feature_sequence, update_max=False):
        
        x, y = self.shapelets_blocks(x)
        y = torch.squeeze(y, 1)
        # if update_max:
        #     max_value = torch.max(y)
        #     self.max = torch.max(torch.tensor([max_value, self.max]))
        # y = 1 - y / self.max
        # y = F.normalize(y, p=2, dim=-1)
        y_min = y.min(dim=-1, keepdim=True)[0]
        y_max = y.max(dim=-1, keepdim=True)[0]
        y = (y - y_min) / (y_max - y_min + 1e-8)
        y = 1 - y
        
        if self.mode == 'fusion':
            # Apply dynamic weighting
            f1_proj = self.proj_1(y)
            f2_proj = self.proj_2(feature_sequence)
            cat_features = torch.cat([f1_proj, f2_proj], dim=-1) # (batch, time_steps, d_model*2)
            fusion_scores = F.softmax(self.fusion_weights(cat_features), dim=-1)
            F1_weighted = fusion_scores[..., 0].unsqueeze(-1) * f1_proj  # (batch, time_steps, d_model)
            F2_weighted = fusion_scores[..., 1].unsqueeze(-1) * f2_proj  # (batch, time_steps, d_model)
            joint_output = F1_weighted + F2_weighted  # (batch, time_steps, d_model)
        else:
            f1_proj = self.proj_1(y)
            f2_proj = self.proj_2(feature_sequence)
            cat_features = torch.cat([f1_proj, f2_proj], dim=-1)
            joint_output = self.concat_layer(cat_features)
        
        batch_size, _, _ = y.shape
        pos_indices = torch.arange(self.transform_seq_len, device=x.device).unsqueeze(0).expand(batch_size, self.transform_seq_len)
        pos_emb = self.positional_emb(pos_indices)
        transformer_out = self.transformer_encoder(joint_output + pos_emb)
        # final_out = self.linear(transformer_out[:, -1, :])
        final_out = self.linear(transformer_out.mean(dim=1))
        return final_out