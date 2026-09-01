"""
PHASE H10 — MODEL K2 DUAL-STREAM SPATIO-TEMPORAL TCN ARCHITECTURE

Separates 187-D spatial pose input into:
1. Spatial Stream (121-D): Dims 0..98 (Coords + Visibilities) + Dims 165..186 (Body Geometry)
2. Motion Stream (66-D): Dims 99..164 (Velocities)

Each stream is processed by a 2-layer 1D Residual TCN (64 channels, dilations [1, 2]).
Outputs are concatenated into 128-D multi-stream features, passed through Temporal Attention Pooling,
and classified via Linear(128->32) -> ReLU -> Dropout(0.5) -> Linear(32->2).
"""

import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=2, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        if out.size(2) != res.size(2):
            out = out[:, :, :res.size(2)]
        return self.relu(out + res)

class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.attn_linear = nn.Linear(in_channels, 1)

    def forward(self, x):
        # x: (B, C, T) where C=128, T=50
        x_perm = x.permute(0, 2, 1) # (B, T, C)
        attn_logits = self.attn_linear(x_perm) # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1) # (B, T, 1)
        pooled = torch.sum(x_perm * attn_weights, dim=1) # (B, C)
        return pooled

class ModelK2_DualStreamTCN(nn.Module):
    """
    Model K2: Dual-Stream Spatio-Temporal Residual TCN.
    Inputs: (B, 50, 187) float32
    Outputs: Logits (B, 2)
    """
    def __init__(
        self,
        spatial_in_dim=121,
        motion_in_dim=66,
        num_channels=None,
        kernel_size=3,
        fc_dim=32,
        dropout_p=0.5,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64]
            
        # 1. Spatial TCN Stream (121-D Input)
        spatial_layers = []
        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            in_ch = spatial_in_dim if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size
            spatial_layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.2)
            )
        self.spatial_tcn = nn.Sequential(*spatial_layers)

        # 2. Motion TCN Stream (66-D Input)
        motion_layers = []
        for i, out_ch in enumerate(num_channels):
            dilation_size = 2 ** i
            in_ch = motion_in_dim if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation_size
            motion_layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=0.2)
            )
        self.motion_tcn = nn.Sequential(*motion_layers)

        # 3. Fusion & Attention
        fused_channels = num_channels[-1] + num_channels[-1] # 64 + 64 = 128
        self.attention_pooling = TemporalAttentionPooling(in_channels=fused_channels)

        # 4. Classification Head
        self.fc1 = nn.Linear(fused_channels, fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, 2)

    def extract_streams(self, x):
        # x: (B, 50, 187)
        # Spatial: dims 0..98 (99) + dims 165..186 (22) = 121
        spatial = torch.cat([x[:, :, 0:99], x[:, :, 165:187]], dim=2) # (B, 50, 121)
        # Motion: dims 99..164 = 66
        motion = x[:, :, 99:165] # (B, 50, 66)
        return spatial, motion

    def forward(self, x):
        # x: (B, 50, 187)
        spatial, motion = self.extract_streams(x)
        
        # Permute for 1D Conv: (B, C, T)
        s_t = spatial.permute(0, 2, 1) # (B, 121, 50)
        m_t = motion.permute(0, 2, 1)  # (B, 66, 50)
        
        feat_s = self.spatial_tcn(s_t) # (B, 64, 50)
        feat_m = self.motion_tcn(m_t)   # (B, 64, 50)
        
        # Fuse streams: (B, 128, 50)
        fused = torch.cat([feat_s, feat_m], dim=1)
        
        # Attention Pooling over sequence length T=50 -> (B, 128)
        pooled = self.attention_pooling(fused)
        
        # Classifier: 128 -> 32 -> 2
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)
