"""
Sensor Behavior Model
=======================
Small feedforward network for classifying driving behavior (e.g.
normal vs aggressive) from numeric windowed sensor scores. Much
smaller than the image CNN -- 7 input features, not 224x224x3 pixels.
"""

import torch.nn as nn


class SensorBehaviorModel(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)