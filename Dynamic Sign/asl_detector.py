import os
import time
import copy
import math

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Set this to control recording duration while keeping 30 model frames:
#   10 fps -> 30 / 10 = 3 seconds per sign
#   15 fps -> 30 / 15 = 2 seconds per sign
CAMERA_FRAME_RATE = 10
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.0


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'best_model.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found at {model_path}. Train the model first.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp_holistic = mp.solutions.holistic

NUM_POSE = 33
NUM_FACE = 468
NUM_HAND = 21
FULL_FEATURE_COUNT = 3 * (NUM_POSE + NUM_FACE + 2 * NUM_HAND)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

COLOR_POSE = (0, 0, 255)
COLOR_FACE = (255, 0, 0)
COLOR_LEFT_HAND = (0, 255, 0)
COLOR_RIGHT_HAND = (128, 0, 128)


DEFAULT_SEQUENCE_LENGTH = 30


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, sequence):
        weights = self.score(sequence).squeeze(-1)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=128):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SignLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.35, num_layers=2, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPool(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_dropout(x)
        output, _ = self.lstm(x)
        return self.classifier(self.attention(output))


class SignGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.35, num_layers=2, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPool(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_dropout(x)
        output, _ = self.gru(x)
        return self.classifier(self.attention(output))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class SignTCNClassifier(nn.Module):
    def __init__(self, input_size, channels, num_classes, dropout=0.30, num_blocks=4, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        dilations = [2 ** (idx % 3) for idx in range(num_blocks)]
        self.blocks = nn.Sequential(*[
            TemporalResidualBlock(channels, dilation=dilation, dropout=dropout)
            for dilation in dilations
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x).transpose(1, 2)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.classifier(pooled)


class SignTransformerClassifier(nn.Module):
    def __init__(
        self, input_size, model_dim, num_classes, dropout=0.30, num_layers=2,
        num_heads=4, ff_mult=4, sequence_length=DEFAULT_SEQUENCE_LENGTH, **kwargs
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, model_dim)
        self.position = PositionalEncoding(model_dim, max_len=max(sequence_length, DEFAULT_SEQUENCE_LENGTH))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPool(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.projection(x)
        x = self.position(x)
        x = self.encoder(x)
        return self.classifier(self.pool(x))


class SignConvGRUClassifier(nn.Module):
    def __init__(self, input_size, conv_channels, hidden_size, num_classes, dropout=0.35, num_layers=1, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            TemporalResidualBlock(conv_channels, dilation=2, dropout=dropout),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionPool(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        output, _ = self.gru(x)
        return self.classifier(self.attention(output))


class InceptionTemporalBlock(nn.Module):
    def __init__(self, channels, bottleneck, dropout=0.25):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
        )
        branch_channels = channels // 4
        self.branches = nn.ModuleList([
            nn.Conv1d(bottleneck, branch_channels, kernel_size=3, padding=1),
            nn.Conv1d(bottleneck, branch_channels, kernel_size=5, padding=2),
            nn.Conv1d(bottleneck, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(bottleneck, branch_channels, kernel_size=1),
        ])
        self.post = nn.Sequential(
            nn.BatchNorm1d(branch_channels * len(self.branches)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(branch_channels * len(self.branches), channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        z = self.pre(x)
        z = torch.cat([branch(z) for branch in self.branches], dim=1)
        return self.activation(x + self.post(z))


class LandmarkI3DClassifier(nn.Module):
    """I3D-inspired temporal inception model for 30-frame landmark clips."""

    def __init__(self, input_size, num_classes, channels=128, bottleneck=64, num_blocks=4, dropout=0.30, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            InceptionTemporalBlock(channels, bottleneck=bottleneck, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.attention = AttentionPool(channels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(channels * 3),
            nn.Dropout(dropout),
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)
        x = self.blocks(self.stem(x)).transpose(1, 2)
        pooled = torch.cat([self.attention(x), x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.classifier(pooled)


def _normalize_adjacency(adjacency):
    adjacency = adjacency + torch.eye(adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def _mediapipe_edges(num_nodes):
    pose_edges = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
    ]
    hand_template = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
        (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    edges = [(i, j) for i, j in pose_edges if i < num_nodes and j < num_nodes]
    for offset in (63, 84):
        edges.extend((offset + i, offset + j) for i, j in hand_template if offset + j < num_nodes)
    for i in range(33, min(63, num_nodes - 1)):
        edges.append((i, i + 1))
    return edges


def build_landmark_adjacency(num_nodes):
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i, j in _mediapipe_edges(num_nodes):
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    return _normalize_adjacency(adjacency)


class TemporalNodeGraphBlock(nn.Module):
    def __init__(self, channels, dropout=0.25, temporal_kernel=3):
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.spatial = nn.Linear(channels, channels)
        self.temporal = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(temporal_kernel, 1), padding=(padding, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, adjacency):
        y = torch.einsum('vw,bctw->bctv', adjacency, x)
        y = self.spatial(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.temporal(y)
        return self.norm((x + y).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)



class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.25):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x + self.net(x))


class GlobalTemporalBranch(nn.Module):
    def __init__(self, input_size, channels=128, dropout=0.30, num_blocks=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.projection = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            DepthwiseTemporalBlock(channels, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.attention = AttentionPool(channels)
        self.out_features = channels * 3

    def forward(self, x):
        x = self.input_norm(x).transpose(1, 2)
        x = self.blocks(self.projection(x)).transpose(1, 2)
        return torch.cat([self.attention(x), x.mean(dim=1), x.max(dim=1).values], dim=1)


class STGCNClassifier(nn.Module):
    """Compact ST-GCN branch plus a full-landmark temporal branch."""

    def __init__(
        self, input_size, num_classes, channels=128, num_blocks=4, dropout=0.30,
        global_channels=128, global_blocks=2, **kwargs
    ):
        super().__init__()
        if input_size % 3 != 0:
            raise ValueError('STGCNClassifier expects xyz landmark triples.')
        self.num_nodes = input_size // 3
        self.input_norm = nn.LayerNorm(input_size)
        self.node_projection = nn.Conv2d(3, channels, kernel_size=1)
        self.register_buffer('adjacency', build_landmark_adjacency(self.num_nodes))
        self.blocks = nn.ModuleList([
            TemporalNodeGraphBlock(channels, dropout=dropout, temporal_kernel=3)
            for _ in range(num_blocks)
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_nodes, self.num_nodes))
            for _ in range(num_blocks)
        ])
        self.graph_attention = AttentionPool(channels)
        self.global_branch = GlobalTemporalBranch(
            input_size, channels=global_channels, dropout=dropout, num_blocks=global_blocks
        )
        fused_features = channels * 3 + self.global_branch.out_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Dropout(dropout),
            nn.Linear(fused_features, max(channels, global_channels)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(channels, global_channels), num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        global_pooled = self.global_branch(x)
        graph_x = self.input_norm(x).view(batch_size, seq_len, self.num_nodes, 3)
        graph_x = graph_x.permute(0, 3, 1, 2).contiguous()
        graph_x = self.node_projection(graph_x)
        for block, importance in zip(self.blocks, self.edge_importance):
            adjacency = self.adjacency * torch.sigmoid(importance)
            graph_x = block(graph_x, adjacency)
        temporal = graph_x.mean(dim=3).transpose(1, 2)
        graph_pooled = torch.cat([
            self.graph_attention(temporal),
            temporal.mean(dim=1),
            temporal.max(dim=1).values,
        ], dim=1)
        return self.classifier(torch.cat([graph_pooled, global_pooled], dim=1))


class TGCNBlock(nn.Module):
    def __init__(self, hidden_features, num_nodes, dropout=0.30):
        super().__init__()
        self.graph = nn.Parameter(torch.eye(num_nodes))
        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        adjacency = torch.softmax(self.graph, dim=-1)
        y = torch.einsum('vw,bwf->bvf', adjacency, x)
        y = self.dropout(torch.tanh(self.norm1(self.fc1(y))))
        y = torch.einsum('vw,bwf->bvf', adjacency, y)
        y = self.dropout(torch.tanh(self.norm2(self.fc2(y))))
        return x + y


class TGCNClassifier(nn.Module):
    """WLASL TGCN-inspired learned graph, fused with full temporal landmarks."""

    def __init__(
        self, input_size, num_classes, hidden_features=192, num_blocks=2, dropout=0.30,
        global_channels=128, global_blocks=2, **kwargs
    ):
        super().__init__()
        if input_size % 3 != 0:
            raise ValueError('TGCNClassifier expects xyz landmark triples.')
        self.num_nodes = input_size // 3
        self.input_norm = nn.LayerNorm(input_size)
        self.node_projection = nn.Linear(DEFAULT_SEQUENCE_LENGTH * 3, hidden_features)
        self.blocks = nn.ModuleList([
            TGCNBlock(hidden_features, self.num_nodes, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.global_branch = GlobalTemporalBranch(
            input_size, channels=global_channels, dropout=dropout, num_blocks=global_blocks
        )
        fused_features = hidden_features * 2 + self.global_branch.out_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Dropout(dropout),
            nn.Linear(fused_features, max(hidden_features, global_channels)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_features, global_channels), num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        global_pooled = self.global_branch(x)
        graph_x = self.input_norm(x).view(batch_size, seq_len, self.num_nodes, 3)
        graph_x = graph_x.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, seq_len * 3)
        graph_x = self.node_projection(graph_x)
        for block in self.blocks:
            graph_x = block(graph_x)
        graph_pooled = torch.cat([graph_x.mean(dim=1), graph_x.max(dim=1).values], dim=1)
        return self.classifier(torch.cat([graph_pooled, global_pooled], dim=1))



class DecoupledGraphBlock(nn.Module):
    def __init__(self, channels, num_nodes, dropout=0.25):
        super().__init__()
        self.semantic_graph = nn.Parameter(torch.eye(num_nodes))
        self.node_gate = nn.Sequential(
            nn.Linear(channels, max(channels // 2, 1)),
            nn.GELU(),
            nn.Linear(max(channels // 2, 1), 1),
        )
        self.spatial = nn.Conv2d(channels, channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, static_adjacency):
        dynamic = torch.softmax(self.semantic_graph, dim=-1)
        node_context = x.mean(dim=2).permute(0, 2, 1)
        gate = torch.sigmoid(self.node_gate(node_context)).squeeze(-1)
        gated_adjacency = static_adjacency.unsqueeze(0) * gate.unsqueeze(1)
        spatial_static = torch.einsum('bvw,bctw->bctv', gated_adjacency, x)
        spatial_dynamic = torch.einsum('vw,bctw->bctv', dynamic, x)
        y = self.spatial(spatial_static + spatial_dynamic)
        y = self.temporal(y)
        return self.norm((x + y).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SLGCNClassifier(nn.Module):
    """OpenHands decoupled-GCN-inspired graph, fused with full temporal landmarks."""

    def __init__(
        self, input_size, num_classes, channels=128, num_blocks=4, dropout=0.30,
        global_channels=128, global_blocks=2, **kwargs
    ):
        super().__init__()
        if input_size % 3 != 0:
            raise ValueError('SLGCNClassifier expects xyz landmark triples.')
        self.num_nodes = input_size // 3
        self.input_norm = nn.LayerNorm(input_size)
        self.node_projection = nn.Conv2d(3, channels, kernel_size=1)
        self.register_buffer('static_adjacency', build_landmark_adjacency(self.num_nodes))
        self.blocks = nn.ModuleList([
            DecoupledGraphBlock(channels, self.num_nodes, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.temporal_attention = AttentionPool(channels)
        self.global_branch = GlobalTemporalBranch(
            input_size, channels=global_channels, dropout=dropout, num_blocks=global_blocks
        )
        fused_features = channels * 3 + self.global_branch.out_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_features),
            nn.Dropout(dropout),
            nn.Linear(fused_features, max(channels, global_channels)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(channels, global_channels), num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        global_pooled = self.global_branch(x)
        graph_x = self.input_norm(x).view(batch_size, seq_len, self.num_nodes, 3)
        graph_x = graph_x.permute(0, 3, 1, 2).contiguous()
        graph_x = self.node_projection(graph_x)
        for block in self.blocks:
            graph_x = block(graph_x, self.static_adjacency)
        temporal = graph_x.mean(dim=3).transpose(1, 2)
        graph_pooled = torch.cat([
            self.temporal_attention(temporal),
            temporal.mean(dim=1),
            temporal.max(dim=1).values,
        ], dim=1)
        return self.classifier(torch.cat([graph_pooled, global_pooled], dim=1))



MODEL_REGISTRY = {
    'bilstm_attention': SignLSTMClassifier,
    'bigru_attention': SignGRUClassifier,
    'tcn': SignTCNClassifier,
    'transformer': SignTransformerClassifier,
    'conv_gru_attention': SignConvGRUClassifier,
    'landmark_i3d': LandmarkI3DClassifier,
    'tgcn': TGCNClassifier,
    'st_gcn': STGCNClassifier,
    'sl_gcn': SLGCNClassifier,
}


def build_model_from_parts(input_size, num_classes, model_type, model_config=None):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model_type {model_type!r}. Available: {sorted(MODEL_REGISTRY)}')
    model_kwargs = copy.deepcopy(model_config or {})
    model_kwargs.pop('model_type', None)
    model_kwargs.pop('num_classes', None)
    return MODEL_REGISTRY[model_type](input_size=input_size, num_classes=num_classes, **model_kwargs)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(input_size, num_classes, model_type, model_config):
    return build_model_from_parts(input_size, num_classes, model_type, model_config)

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    label_to_id = checkpoint['label_to_id']
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    model_config = checkpoint['model_config']
    input_size = int(checkpoint['input_size'])
    model_type = checkpoint['model_type']
    num_classes = int(model_config['num_classes'])

    if num_classes != len(label_to_id):
        raise ValueError(f'Model has {num_classes} classes, but label_to_id has {len(label_to_id)} labels.')

    model = build_model(
        input_size=input_size,
        num_classes=num_classes,
        model_type=model_type,
        model_config=model_config,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    feature_mean = np.asarray(checkpoint['feature_mean'], dtype=np.float32)
    feature_std = np.asarray(checkpoint['feature_std'], dtype=np.float32)
    sequence_length = int(checkpoint['sequence_length'])

    landmark_selection = checkpoint.get('landmark_selection')
    selected_feature_indices = None
    if landmark_selection is not None:
        selected_feature_indices = np.asarray(
            landmark_selection['selected_feature_indices'],
            dtype=np.int64,
        )
        selected_feature_count = int(landmark_selection.get('selected_feature_count', len(selected_feature_indices)))
        if selected_feature_count != input_size:
            raise ValueError(
                f'Checkpoint expects {input_size} features, but landmark_selection '
                f'describes {selected_feature_count}.'
            )
        if len(selected_feature_indices) != input_size:
            raise ValueError(
                f'Checkpoint expects {input_size} features, but selected_feature_indices '
                f'contains {len(selected_feature_indices)}.'
            )
        if selected_feature_indices.min(initial=0) < 0 or selected_feature_indices.max(initial=0) >= FULL_FEATURE_COUNT:
            raise ValueError('selected_feature_indices contains values outside the full landmark feature vector.')

    if len(feature_mean) != input_size or len(feature_std) != input_size:
        raise ValueError(
            f'Normalization stats must match model input size {input_size}; '
            f'got mean={len(feature_mean)}, std={len(feature_std)}.'
        )

    return model, id_to_label, feature_mean, feature_std, sequence_length, input_size, selected_feature_indices, model_type


(
    model,
    id_to_label,
    feature_mean,
    feature_std,
    sequence_length,
    model_input_size,
    selected_feature_indices,
    loaded_model_type,
) = load_checkpoint(model_path)


def landmarks_to_array(landmarks, expected_count):
    arr = np.zeros((expected_count, 3), dtype=np.float32)
    if landmarks is None:
        return arr

    for i, landmark in enumerate(landmarks.landmark[:expected_count]):
        arr[i] = [landmark.x, landmark.y, landmark.z]
    return arr


def normalize_landmarks(pose, face, left_hand, right_hand):
    all_points = np.concatenate([pose, face, left_hand, right_hand], axis=0)
    detected_mask = np.any(all_points != 0, axis=1)

    left_shoulder = pose[LEFT_SHOULDER]
    right_shoulder = pose[RIGHT_SHOULDER]

    shoulders_detected = np.any(left_shoulder) and np.any(right_shoulder)
    if shoulders_detected:
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if scale > 1e-6:
            all_points[detected_mask] = (all_points[detected_mask] - center) / scale

    return all_points.reshape(-1).astype(np.float32)


def standardize_sequence(sequence):
    sequence = sequence.copy().astype(np.float32)
    mask = sequence != 0
    sequence[mask] = ((sequence - feature_mean) / feature_std)[mask]
    return sequence


def select_landmark_features(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f'Expected a 2D landmark sequence, got shape {sequence.shape}')

    if selected_feature_indices is None:
        if sequence.shape[1] != model_input_size:
            raise ValueError(
                f'Checkpoint does not include landmark_selection metadata, and sequence has '
                f'{sequence.shape[1]} features instead of model input size {model_input_size}.'
            )
        return sequence

    if sequence.shape[1] != FULL_FEATURE_COUNT:
        raise ValueError(
            f'Feature selection expects full {FULL_FEATURE_COUNT}-feature frames before selection, '
            f'got {sequence.shape[1]}.'
        )

    return sequence[:, selected_feature_indices]


def extract_landmarks(frame, holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    pose = landmarks_to_array(results.pose_landmarks, NUM_POSE)
    face = landmarks_to_array(results.face_landmarks, NUM_FACE)
    left_hand = landmarks_to_array(results.left_hand_landmarks, NUM_HAND)
    right_hand = landmarks_to_array(results.right_hand_landmarks, NUM_HAND)

    features = normalize_landmarks(pose, face, left_hand, right_hand)
    return features, results


def draw_landmark_group(frame, landmark_list, connections, color, radius=4, thickness=2):
    if landmark_list is None:
        return frame

    height, width = frame.shape[:2]
    points = []

    for landmark in landmark_list.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), radius, color, -1)

    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, thickness)

    return frame


def draw_face_dots(frame, landmark_list):
    if landmark_list is None:
        return frame

    height, width = frame.shape[:2]
    for landmark in landmark_list.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, COLOR_FACE, -1)

    return frame


def draw_results(frame, results):
    if results is None:
        return frame

    draw_face_dots(frame, results.face_landmarks)
    draw_landmark_group(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, COLOR_POSE, radius=4, thickness=2)
    draw_landmark_group(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, COLOR_LEFT_HAND, radius=4, thickness=2)
    draw_landmark_group(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, COLOR_RIGHT_HAND, radius=4, thickness=2)
    return frame


def predict_sign(landmark_sequence):
    if len(landmark_sequence) != sequence_length:
        raise ValueError(f'Expected {sequence_length} frames, got {len(landmark_sequence)}')

    sequence = np.asarray(landmark_sequence, dtype=np.float32)
    sequence = select_landmark_features(sequence)
    sequence = standardize_sequence(sequence)
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sequence_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_idx].item())

    if confidence < CONFIDENCE_THRESHOLD:
        return 'Unknown', confidence

    return id_to_label[predicted_idx], confidence


def main():
    if CAMERA_FRAME_RATE <= 0:
        raise ValueError('CAMERA_FRAME_RATE must be greater than 0.')

    capture_seconds = sequence_length / CAMERA_FRAME_RATE
    print(f'Loaded model: {model_path}')
    print(f'Model type: {loaded_model_type}')
    print(f'Classes: {id_to_label}')
    print(f'Model input features: {model_input_size}')
    if selected_feature_indices is not None:
        print(f'Applying selected landmark features: {len(selected_feature_indices)} of {FULL_FEATURE_COUNT}')
    print(f'Capturing {sequence_length} frames at {CAMERA_FRAME_RATE} fps ({capture_seconds:.1f}s per sign).')
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return

    landmark_buffer = []
    last_sample_time = 0.0
    current_prediction = 'Sign: waiting...'
    last_results = None
    sign_start_time = time.time()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Could not read frame.')
                break

            now = time.time()
            should_sample = now - last_sample_time >= 1.0 / CAMERA_FRAME_RATE

            if should_sample:
                last_sample_time = now
                landmarks, results = extract_landmarks(frame, holistic)
                last_results = results
                landmark_buffer.append(landmarks)

                if len(landmark_buffer) >= sequence_length:
                    elapsed = now - sign_start_time
                    sign, confidence = predict_sign(landmark_buffer[:sequence_length])
                    current_prediction = f'Sign: {sign} ({confidence * 100:.1f}%)'
                    print(f'Captured {sequence_length} frames in {elapsed:.2f}s')
                    landmark_buffer = []
                    sign_start_time = time.time()

            frame = draw_results(frame, last_results)

            progress = len(landmark_buffer)
            cv2.putText(
                frame,
                f'Frames: {progress}/{sequence_length} | {CAMERA_FRAME_RATE} fps | {capture_seconds:.1f}s',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                current_prediction,
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow('ASL Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
