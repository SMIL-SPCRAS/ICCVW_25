# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .help_layers import TransformerEncoderLayer, MambaBlock 

class ProbabilityFusion(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(ProbabilityFusion, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))

    def forward(self, pred):
        normalized_weights = torch.softmax(self.weights, dim=0)
        weighted_matrix = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix, normalized_weights
    
class ProbabilityFusion_v2(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7, hidden_dim=32):
        super(ProbabilityFusion_v2, self).__init__()
        # More complex model for weights
        self.mlp = nn.Sequential(
            nn.Linear(num_matrices * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_matrices * num_classes)
        )
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))
        
    def forward(self, pred):
        # Context-dependent weights
        context = torch.cat(pred, dim=-1)
        dynamic_weights = self.mlp(context).view_as(self.weights)
        
        # Combination of static and dynamic weights
        combined_weights = self.weights + dynamic_weights
        normalized_weights = torch.softmax(combined_weights, dim=0)
        
        weighted_matrix = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix, normalized_weights
    
class ProbabilityFusion_v3(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(ProbabilityFusion_v3, self).__init__()
        self.query = nn.Parameter(torch.randn(num_classes, 1))
        self.keys = nn.Parameter(torch.randn(num_matrices, num_classes))
        self.num_classes = num_classes
        
    def forward(self, pred):
        # Compute attention between query and keys
        attn_scores = torch.matmul(self.keys, self.query) / (self.num_classes ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        # Apply attention weights
        weighted_matrix = sum(mat * attn_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix, attn_weights
    
class ProbabilityFusion_v4(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(ProbabilityFusion_v4, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))
        self.res_weight = nn.Parameter(torch.tensor(0.1))  # learnable residual weight
        
    def forward(self, pred):
        normalized_weights = torch.softmax(self.weights, dim=0)
        simple_fusion = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        
        # Add residual connection (simple average)
        residual = sum(pred) / len(pred)
        weighted_matrix = (1 - torch.sigmoid(self.res_weight)) * simple_fusion + torch.sigmoid(self.res_weight) * residual
        
        return weighted_matrix, normalized_weights
    
class ProbabilityFusion_v5(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7):
        super(ProbabilityFusion_v5, self).__init__()
        self.weights = nn.Parameter(torch.rand(num_matrices, num_classes))
        self.temperature = nn.Parameter(torch.tensor(1.0))  # learnable temperature
        
    def forward(self, pred):
        # Apply temperature to softmax
        normalized_weights = torch.softmax(self.weights / self.temperature.clamp(min=0.1), dim=0)
        weighted_matrix = sum(mat * normalized_weights[i] for i, mat in enumerate(pred))
        return weighted_matrix, normalized_weights
    
class ProbabilityFusion_v6(nn.Module):
    def __init__(self, num_matrices=2, num_classes=7, num_heads=3):
        super(ProbabilityFusion_v6, self).__init__()
        self.num_heads = num_heads
        self.head_weights = nn.Parameter(torch.rand(num_heads, num_matrices, num_classes))
        self.head_combine = nn.Parameter(torch.rand(num_heads))
        
    def forward(self, pred):
        # Each head computes its own weights
        head_normalized = torch.softmax(self.head_weights, dim=1)
        
        # Compute weighted matrices for each head
        head_outputs = []
        for h in range(self.num_heads):
            head_outputs.append(sum(mat * head_normalized[h,i] for i, mat in enumerate(pred)))
        
        # Combine head outputs
        combine_weights = torch.softmax(self.head_combine, dim=0)
        weighted_matrix = sum(h_out * combine_weights[h] for h, h_out in enumerate(head_outputs))
        
        return weighted_matrix, (head_normalized, combine_weights)

class VideoMamba(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, mamba_d_state=8, mamba_ker_size=3, mamba_layer_number=2, d_discr=None, dropout=0.1, seg_len=20, out_features=128, num_classes=7, device='cpu'):
        super(VideoMamba, self).__init__()

        mamba_par = {
            'd_input': hidden_dim,
            'd_model': hidden_dim,
            'd_state': mamba_d_state,
            'd_discr': d_discr,
            'ker_size': mamba_ker_size,
            'dropout': dropout,
            'device': device
        }

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.mamba = nn.ModuleList([
            MambaBlock(**mamba_par) for _ in range(mamba_layer_number)
        ])

        self._calculate_classifier_input_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    def forward(self, sequences, features=False):
        sequences = self.image_proj(sequences)

        for i in range(len(self.mamba)):
            att_sequences, _ = self.mamba[i](sequences)
            sequences = sequences + att_sequences

        sequences_poll = self._pool_features(sequences)

        out = self.classifier(sequences_poll)

        if features:
            return {'prob': out,
                    'features': sequences_poll}
        else:
            return out
    
    def _calculate_classifier_input_dim(self):
        """Calculates input feature size for classifier"""
        # Test pass through pooling with dummy data
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)

        video_pool = self._pool_features(dummy_video)

        self.classifier_input_dim = video_pool.size(1)
    
    def _pool_features(self, x):
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]
        return mean_temp
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

class VideoFormer(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128,
                num_transformer_heads=2, positional_encoding=True, dropout=0.1, tr_layer_number=2, seg_len=20, out_features=128, num_classes=7):
        super(VideoFormer, self).__init__()

        self.seg_len = seg_len
        self.hidden_dim = hidden_dim

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding
            ) for _ in range(tr_layer_number)
        ])

        self._calculate_classifier_input_dim()

        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_classes)
        )

        self._init_weights()

    def forward(self, sequences):
        sequences = self.image_proj(sequences)

        for i in range(len(self.transformer)):
            att_sequences = self.transformer[i](sequences, sequences, sequences)
            sequences = sequences + att_sequences

        sequences_poll = self._pool_features(sequences)

        return self.classifier(sequences_poll)
    
    def _calculate_classifier_input_dim(self):
        """Calculates input feature size for classifier"""
        # Test pass through pooling with dummy data
        dummy_video = torch.randn(1, self.seg_len, self.hidden_dim)

        video_pool = self._pool_features(dummy_video)

        self.classifier_input_dim = video_pool.size(1)
    
    def _pool_features(self, x):
        mean_temp = x.mean(dim=1)  # [batch, hidden_dim]
        return mean_temp
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)