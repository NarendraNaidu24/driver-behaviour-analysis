"""
Multi-Task Driver Behavior Model -- Domain-Specific BatchNorm
=================================================================
Extends task-specific BatchNorm one level further: BatchNorm statistics
are now specific to each (task, dataset) DOMAIN, not just each task.

Why: task-specific BN fixed interference BETWEEN tasks (distraction vs
drowsiness). But once distraction itself combined two visually distinct
datasets (State Farm, DMD), the SAME interference problem reappeared
WITHIN the distraction task, because both datasets still shared one set
of BatchNorm stats. Domain-specific BN fixes this by giving every
(task, dataset) combination its own normalization, while classification
heads remain shared per TASK -- so the model still learns one general
concept of "texting" etc., it just normalizes each data source properly
before doing so.

Domain naming convention: f"{task}::{dataset}", e.g.
"distraction::state_farm", "distraction::dmd", "drowsiness::yawdd".
"""

import torch
import torch.nn as nn
import torchvision.models as models


def make_domain(task: str, dataset: str) -> str:
    return f"{task}::{dataset}"


def task_from_domain(domain: str) -> str:
    return domain.split("::", 1)[0]


class DomainSpecificBatchNorm2d(nn.Module):
    """Holds one independent BatchNorm2d per (task, dataset) domain."""
    def __init__(self, num_features, domains):
        super().__init__()
        self.bns = nn.ModuleDict({d: nn.BatchNorm2d(num_features) for d in domains})
        self.active_domain = domains[0]

    def set_domain(self, domain):
        self.active_domain = domain

    def forward(self, x):
        return self.bns[self.active_domain](x)


def _replace_batchnorm_with_domain_specific(module, domains):
    created = []
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new_bn = DomainSpecificBatchNorm2d(child.num_features, domains)
            setattr(module, name, new_bn)
            created.append(new_bn)
        else:
            created.extend(_replace_batchnorm_with_domain_specific(child, domains))
    return created


class MultiTaskDriverModel(nn.Module):
    def __init__(self, num_classes: dict, domains: list, backbone: str = "resnet18",
                 pretrained: bool = True):
        """
        num_classes : dict like {"distraction": 8, "drowsiness": 4}
                       (defines the task heads -- shared across datasets
                       within a task)
        domains     : list of domain strings, e.g.
                       ["distraction::state_farm", "distraction::dmd",
                        "drowsiness::yawdd"] (defines BatchNorm groups --
                       one per task+dataset combination)
        backbone    : "resnet18" (fast) or "efficientnet_b0" (stronger)
        """
        super().__init__()
        self.domains = domains
        self.num_classes = num_classes

        if backbone == "resnet18":
            base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.trunk = base
        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.trunk = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.domain_bn_modules = _replace_batchnorm_with_domain_specific(self.trunk, domains)
        print(f"[model] replaced {len(self.domain_bn_modules)} BatchNorm layers with "
              f"domain-specific versions for domains: {domains}")

        if backbone == "resnet18":
            self._freezable_layers = [base.conv1, base.bn1, base.layer1, base.layer2, base.layer3]
        else:
            self._freezable_layers = []

        self.feat_dim = feat_dim
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )
            for task, n_classes in num_classes.items()
        })

    def freeze_early_layers(self):
        for layer in self._freezable_layers:
            for param in layer.parameters():
                param.requires_grad = False
        print("[model] froze early backbone layers (conv1, bn1, layer1-3); "
              "training layer4 + heads only")

    def _set_active_domain(self, domain):
        for bn in self.domain_bn_modules:
            bn.set_domain(domain)

    def forward(self, x, task: str, dataset: str = None):
        """
        `task` selects which classification head to use (required).
        `dataset` selects which BatchNorm domain to use. If omitted,
        falls back to the first registered domain for this task (used
        e.g. when running inference on genuinely novel data where the
        source dataset isn't known -- not ideal, but a reasonable
        default; OOD detection should be used to sanity-check such
        predictions regardless).
        """
        if dataset is not None:
            domain = make_domain(task, dataset)
        else:
            matching = [d for d in self.domains if task_from_domain(d) == task]
            if not matching:
                raise ValueError(f"No domain registered for task '{task}'")
            domain = matching[0]

        self._set_active_domain(domain)
        feats = self.trunk(x)
        return {task: self.heads[task](feats)}

    def get_features(self, x, task: str, dataset: str = None):
        """Same domain-selection logic as forward(), but returns raw
        features (before the classification head) for OOD detection."""
        if dataset is not None:
            domain = make_domain(task, dataset)
        else:
            matching = [d for d in self.domains if task_from_domain(d) == task]
            if not matching:
                raise ValueError(f"No domain registered for task '{task}'")
            domain = matching[0]
        self._set_active_domain(domain)
        return self.trunk(x)