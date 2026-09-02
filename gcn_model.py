"""
gcn_model.py — a Graph Convolutional Network over the 21 hand landmarks,
using the fixed MediaPipe hand skeleton as the graph structure.

Why a fixed graph makes this simple: every sample has the SAME 21 joints
connected the SAME way (wrist -> knuckles -> fingertips). Unlike general
GCN libraries built for variable-size graphs, we can precompute one
adjacency matrix once and reuse it for every sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The 21-landmark hand skeleton, as MediaPipe defines it.
# 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky.
# Sanity check if you have mediapipe installed: list(mp.solutions.hands.HAND_CONNECTIONS)
# should match this set.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle (+ palm arc from index base)
    (9, 13), (13, 14), (14, 15), (15, 16), # ring (+ palm arc from middle base)
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky (+ palm arc from ring base)
    (0, 17),                                # wrist -> pinky base (closes the palm)
]
NUM_NODES = 21


def build_normalized_adjacency():
    """Builds A_hat = D^-1/2 (A + I) D^-1/2 — the standard GCN propagation
    matrix (Kipf & Welling). Adding self-loops (+I) means a node's own
    features are preserved, not just its neighbors'."""
    A = torch.zeros(NUM_NODES, NUM_NODES)
    for i, j in HAND_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0  # undirected
    A = A + torch.eye(NUM_NODES)  # self-loops

    degree = A.sum(dim=1)
    D_inv_sqrt = torch.diag(degree.pow(-0.5))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_hat):
        # x: (batch, 21, in_dim) — propagate each node's features to its
        # neighbors (weighted by A_hat), THEN transform with a learned linear layer.
        x = torch.einsum("ij,bjf->bif", A_hat, x)
        return self.linear(x)


class HandGCN(nn.Module):
    def __init__(self, num_classes, node_feat_dim=3, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.register_buffer("A_hat", build_normalized_adjacency())
        self.gcn1 = GCNLayer(node_feat_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Flatten (21 nodes x hidden_dim) into one vector, preserving WHICH
        # node contributed what — mean-pooling erased this identity, which
        # is exactly the information hand-letter classification needs most.
        self.fc = nn.Linear(NUM_NODES * hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, 21, 3)
        x = F.relu(self.gcn1(x, self.A_hat))
        x = F.relu(self.gcn2(x, self.A_hat))
        x = F.relu(self.gcn3(x, self.A_hat))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)  # (batch, 21*hidden_dim) — flatten, don't average
        return self.fc(x)
