from random import randint
from typing import List, Union


class LVQ:
    def __init__(self, codebook_size: int, features_count: int, labels_count: int):
        """LVQ algorithm implementation.

        Allows to create, train and use a codebook. Codebook is a list of vectors.
        Vector is a list of weights/features and a label.
        """

        self.features_count = features_count
        self.codebook = [
            [0] * features_count + [i % labels_count + 1] for i in range(codebook_size)
        ]

    @staticmethod
    def vector_euclidean_distance(a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance for all the features (ignore label)."""

        *features_a, _ = a
        *features_b, _ = b
        return sum([(v - x) ** 2 for v, x in zip(features_a, features_b)]) ** 0.5

    def get_best_matching_vector(self, input_vector: List[float]) -> List[float]:
        distances = []
        for vector in self.codebook:
            distances.append(self.vector_euclidean_distance(vector, input_vector))

        closest_vector = self.codebook[distances.index(min(distances))]
        return closest_vector

    def predict(self, input_features: List[float]) -> int:
        return self.get_best_matching_vector(input_features + [None])[-1]

    def train_codebook(
        self, train_vectors: List[List[float]], learning_rate: float, epochs: int
    ) -> None:
        for epoch in range(epochs):
            sse = 0.0
            for t_vector in train_vectors:
                b_vector = self.get_best_matching_vector(t_vector)

                for idx in range(self.features_count):
                    error = t_vector[idx] - b_vector[idx]
                    sse += error**2

                    if t_vector[-1] == b_vector[-1]:
                        b_vector[idx] += learning_rate * error
                    else:
                        b_vector[idx] -= learning_rate * error
            print(f"> epoch {epoch:>5}, error {round(sse,3):>5}")
