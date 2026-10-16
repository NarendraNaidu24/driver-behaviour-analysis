import random
import os
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Make runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_taxonomy(path: str = "configs/taxonomy.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_label_encoders(taxonomy: dict) -> dict:
    """
    Returns {task_name: {class_name: index}} for each task defined
    in the taxonomy. Used to turn string labels in manifests into
    integer targets for training.
    """
    encoders = {}
    for task, spec in taxonomy["tasks"].items():
        classes = spec["classes"]
        encoders[task] = {c: i for i, c in enumerate(classes)}
    return encoders


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)