"""
Temporal Multi-Task Driver Behavior Model
============================================
Extends MultiTaskDriverModel to classify short video CLIPS instead of
single frames, to capture MOTION -- e.g. hand moving toward the ear
(phone_call) vs down toward the lap (texting) vs to the side
(reaching_behind), which look identical in a single still frame but
differ in motion over time.

Reuses the exact same CNN trunk (with domain-specific BatchNorm) frame
by frame, then aggregates the resulting sequence of feature vectors
with a small LSTM before the same classification heads. This means
everything already built (domain-specific BN, OOD detection on
features) still works -- we're adding a temporal aggregation step, not
replacing the visual feature extractor.
"""

import torch
import torch.nn as nn

from src.models.multitask_model import MultiTaskDriverModel


class TemporalMultiTaskModel(nn.Module):
    def __init__(self, num_classes: dict, domains: list, backbone: str = "resnet18",
                 pretrained: bool = True, lstm_hidden: int = 256):
        super().__init__()
        self.frame_model = MultiTaskDriverModel(num_classes, domains, backbone=backbone,
                                                  pretrained=pretrained)
        feat_dim = self.frame_model.feat_dim

        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hidden,
                              batch_first=True, bidirectional=False)

        self.temporal_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(lstm_hidden, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )
            for task, n_classes in num_classes.items()
        })

    def freeze_early_layers(self):
        self.frame_model.freeze_early_layers()

    def get_clip_features(self, x, task: str, dataset: str = None):
        """
        x: (batch, time, channels, height, width)
        Returns the LSTM's final hidden state per clip: (batch, lstm_hidden)
        """
        batch, time_steps, c, h, w = x.shape
        x_flat = x.view(batch * time_steps, c, h, w)

        frame_feats = self.frame_model.get_features(x_flat, task=task, dataset=dataset)
        frame_feats = frame_feats.view(batch, time_steps, -1)

        _, (h_n, _) = self.lstm(frame_feats)
        clip_feat = h_n[-1]  # (batch, lstm_hidden)
        return clip_feat

    def forward(self, x, task: str, dataset: str = None):
        clip_feat = self.get_clip_features(x, task=task, dataset=dataset)
        return {task: self.temporal_heads[task](clip_feat)}