"""
Out-of-Distribution (OOD) / Anomaly Detector -- Per-Class Thresholds
========================================================================
Extends the original Mahalanobis OOD detector with PER-CLASS thresholds
instead of one global threshold shared across all classes.

Why: confirmed via evaluation that a single global threshold unfairly
penalizes smaller/less-represented classes (adjusting_radio_or_console,
other_distraction had 100% precision but only 7-11% recall -- genuine
examples were being flagged as "unrecognized" because their natural
feature-space spread differs from more common classes like
normal_driving, which dominated the global threshold calibration).

Now: during calibration, we track each validation sample's distance to
its OWN TRUE class's centroid (not just the nearest centroid overall),
and set a threshold PER CLASS. At inference, once we find the nearest
centroid to a new sample, we compare against THAT class's own
threshold, not a shared one.
"""

import torch
import numpy as np


class MahalanobisOODDetector:
    def __init__(self):
        self.class_means = None
        self.inv_covariance = None
        self.thresholds = None
        self.default_threshold = None

    def fit(self, model, dataloader, task, device, num_classes, input_key="image"):
        model.eval()
        features_by_class = {c: [] for c in range(num_classes)}

        with torch.no_grad():
            for batch in dataloader:
                images = batch[input_key].to(device)
                labels = batch["label"]
                feats = model.get_features(images, task=task).cpu().numpy()
                for f, l in zip(feats, labels.tolist()):
                    features_by_class[l].append(f)

        self.class_means = {}
        all_centered = []
        for c, feats in features_by_class.items():
            if len(feats) == 0:
                continue
            feats = np.stack(feats)
            mean = feats.mean(axis=0)
            self.class_means[c] = mean
            all_centered.append(feats - mean)

        all_centered = np.concatenate(all_centered, axis=0)
        cov = np.cov(all_centered, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_covariance = np.linalg.inv(cov)

        print(f"[ood] fitted centroids for {len(self.class_means)} classes "
              f"from {sum(len(f) for f in features_by_class.values())} samples")

    def calibrate_threshold(self, model, dataloader, task, device, percentile=95, input_key="image"):
        distances_by_class = {c: [] for c in self.class_means}
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[input_key].to(device)
                labels = batch["label"].tolist()
                feats = model.get_features(images, task=task).cpu().numpy()
                for f, label in zip(feats, labels):
                    if label not in self.class_means:
                        continue
                    diff = f - self.class_means[label]
                    dist = diff @ self.inv_covariance @ diff.T
                    distances_by_class[label].append(dist)

        self.thresholds = {}
        all_distances = []
        for c, dists in distances_by_class.items():
            if len(dists) >= 5:
                self.thresholds[c] = float(np.percentile(dists, percentile))
            all_distances.extend(dists)

        self.default_threshold = float(np.percentile(all_distances, percentile))

        print(f"[ood] calibrated per-class thresholds at {percentile}th percentile:")
        for c, t in self.thresholds.items():
            print(f"    class {c}: {t:.2f} (n={len(distances_by_class[c])})")
        print(f"[ood] default/fallback threshold (classes with too little data): "
              f"{self.default_threshold:.2f}")

    def _nearest_class_and_distance(self, feature_vec):
        best_class, best_dist = None, None
        for c, mean in self.class_means.items():
            diff = feature_vec - mean
            d = diff @ self.inv_covariance @ diff.T
            if best_dist is None or d < best_dist:
                best_dist, best_class = d, c
        return best_class, best_dist

    def is_out_of_distribution(self, feature_vec):
        """Returns (is_ood: bool, distance: float, nearest_class: int)"""
        if self.thresholds is None:
            raise RuntimeError("Call calibrate_threshold() before using this.")
        nearest_class, dist = self._nearest_class_and_distance(feature_vec)
        threshold = self.thresholds.get(nearest_class, self.default_threshold)
        return dist > threshold, dist, nearest_class

    def save(self, path):
        np.savez(path, class_means=self.class_means, inv_covariance=self.inv_covariance,
                  thresholds=self.thresholds, default_threshold=self.default_threshold)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.class_means = data["class_means"].item()
        self.inv_covariance = data["inv_covariance"]
        self.thresholds = data["thresholds"].item()
        self.default_threshold = float(data["default_threshold"])