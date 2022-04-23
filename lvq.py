from random import seed, shuffle, uniform
from typing import List, Union
from tqdm import tqdm


class LVQ:
    """LVQ algorithm implementation.

    Allows to create, train and use a codebook. Codebook is a list of vectors.
    Vector is a list of weights/features and a label.
    """

    def __init__(
        self,
        codebook_size: int,
        features_count: int,
        labels_count: int,
        codebook_init_method: str = "zeros",
        codebook_init_dataset: List[float] = None,
    ):

        self.codebook_size = codebook_size
        self.features_count = features_count
        self.labels_count = labels_count

        assert codebook_init_method in (
            "zeros",
            "sample",
            "random",
        ), "Currently supported codebook initialization methods are: zeros, sample, random"
        if codebook_init_method == "sample":
            assert (
                codebook_init_dataset is not None
            ), "Dataset is needed for sample codebook initialization"
            assert (
                len(codebook_init_dataset) >= codebook_size
            ), "Not enough samples in the dataset"

        if codebook_init_method == "zeros":
            self._init_codebook_zeros()
        elif codebook_init_method == "sample":
            self._init_codebook_sample(codebook_init_dataset)
        elif codebook_init_method == "random":
            self._init_codebook_random()

    def _init_codebook_zeros(self) -> None:
        """Initialize codebook with zeros for all the features.

        Tries to take the same amout of samples for each label.
        """
        self.codebook = [
            [0] * self.features_count + [i % self.labels_count + 1]
            for i in range(self.codebook_size)
        ]

    def _init_codebook_sample(self, dataset: List[List[float]]) -> None:
        """Initialize codebook based on sample dataset.

        Takes some samples from the dataset to initialize the codebook.
        Tries to take the same amout of samples for each label.
        """
        label_split = {label + 1: [] for label in range(self.labels_count)}
        for vector in dataset:
            label_split[vector[-1]].append(vector)

        self.codebook = []
        idx = 1
        while len(self.codebook) < self.codebook_size:
            if len(label_split[idx]) > 0:
                self.codebook.append(label_split[idx].pop())
            idx = (idx + 1) % self.labels_count + 1

    def _init_codebook_random(self) -> None:
        """Initialize the codebook with random values bewteen 0 and 1.

        Tries to take the same amout of samples for each label.
        """
        self.codebook = [
            [uniform(0, 1) for _ in range(self.features_count)]
            + [i % self.labels_count + 1]
            for i in range(self.codebook_size)
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
        self, train_vectors: List[List[float]], base_learning_rate: float, epochs: int
    ) -> None:
        progress = tqdm(
            range(epochs),
            unit="epochs",
            ncols=100,
            bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
        )
        for epoch in progress:
            sse = 0.0
            learning_rate = self.linear_decay(base_learning_rate, epoch, epochs)
            for t_vector in train_vectors:
                b_vector = self.get_best_matching_vector(t_vector)

                for idx in range(self.features_count):
                    error = t_vector[idx] - b_vector[idx]
                    sse += error**2

                    if t_vector[-1] == b_vector[-1]:
                        b_vector[idx] += learning_rate * error
                    else:
                        b_vector[idx] -= learning_rate * error
            progress.set_postfix(sse=sse)

    @staticmethod
    def linear_decay(base_rate: float, current_epoch: int, total_epochs: int) -> float:
        return base_rate * (1.0 - (current_epoch / total_epochs))


def cross_validate(
    # Validation params
    dataset: List[List[float]],
    fold_count: int,
    learning_rate: float,
    epochs: int,
    # Codebook params
    codebook_size: int,
    features_count: int,
    labels_count: int,
    codebook_init_method: str = "zeros",
    codebook_init_dataset: List[float] = None,
) -> List[float]:

    dataset_copy = dataset.copy()

    shuffle(dataset_copy)

    fold_size = len(dataset) // fold_count
    folds = [
        dataset_copy[idx : idx + fold_size] for idx in range(0, len(dataset), fold_size)
    ]

    scores = []
    for test_vectors in folds:
        train_vectors = folds.copy()
        train_vectors.remove(test_vectors)
        train_vectors = [item for fold in train_vectors for item in fold]

        model = LVQ(
            codebook_size=codebook_size,
            features_count=features_count,
            labels_count=labels_count,
            codebook_init_method=codebook_init_method,
            codebook_init_dataset=codebook_init_dataset,
        )
        model.train_codebook(train_vectors, learning_rate, epochs)

        correct = 0
        for vector in test_vectors:
            *features, label = vector
            prediction = model.predict(features)
            if prediction == label:
                correct += 1
        scores.append(correct / len(test_vectors))

    return scores
