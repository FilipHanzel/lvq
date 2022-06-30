import os
import csv
from typing import List, Dict, Any


def encode_labels(dataset: List[List[Any]]) -> (Dict[Any, int], List[List[Any]]):
    """Encode dataset labels. Return label: encoding mapping and mapped dataset."""

    mapping = {label: idx for idx, label in enumerate({label for *_, label in dataset})}
    mapped_dataset = [[*features, mapping[label]] for *features, label in dataset]

    return mapping, mapped_dataset


def normalize(dataset: List[List[float]]) -> List[List[float]]:
    feature_count = len(dataset[0]) - 1

    total_min = [None] * feature_count
    total_max = [None] * feature_count
    for *features, _ in dataset:
        for idx, feature in enumerate(features):
            if total_min[idx] is None or total_min[idx] > feature:
                total_min[idx] = feature
            if total_max[idx] is None or total_max[idx] < feature:
                total_max[idx] = feature

    normalized_dataset = []
    for *features, label in dataset:
        normalized_dataset.append(
            [
                (feature - min_) / (max_ - min_)
                for feature, min_, max_ in zip(features, total_min, total_max)
            ]
            + [label]
        )
    return normalized_dataset
