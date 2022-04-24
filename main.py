import random
from pprint import pprint
import argparse

import lvq
from data import normalize, encode_labels
from data import load_ionosphere_data, load_iris_data


def pprint_vector(vector):
    vector = [str(round(value, 3)) for value in vector]
    if len(vector) > 6:
        vector = [*vector[:3], "...", *vector[-3:]]
    print("[" + ", ".join([f"{value:>7}" for value in vector]) + "]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Examples and tests of LVQ algorithm pure python implementation."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        action="store",
        choices=["iris", "ionosphere"],
        default="iris",
        dest="dataset",
    )
    parser.add_argument("--seed", "-r", action="store", default=None, dest="seed")
    parser.add_argument(
        "--codebook-size",
        "-s",
        action="store",
        type=int,
        required=True,
        dest="codebook_size",
    )
    parser.add_argument(
        "--codebook-init-method",
        "-i",
        action="store",
        choices=["zeros", "random", "sample"],
        default="sample",
        dest="codebook_init_method",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        action="store",
        type=float,
        default=0.01,
        dest="learning_rate",
    )
    parser.add_argument(
        "--learning-rate-decay",
        "-lrd",
        action="store",
        type=str,
        default=None,
        dest="learning_rate_decay",
    )
    parser.add_argument(
        "--epochs", "-e", action="store", type=int, default=100, dest="epochs"
    )
    parser.add_argument(
        "--cross-validation-folds",
        "-f",
        action="store",
        type=int,
        default=5,
        dest="cross_validation_folds",
    )

    args = parser.parse_args()

    if args.dataset == "iris":
        labels_mapping, dataset = load_iris_data()
    elif args.dataset == "ionosphere":
        labels_mapping, dataset = load_ionosphere_data()

    print("Label mapping:")
    for label in labels_mapping:
        print(f"\t{label + ':':20} {labels_mapping[label]}")

    if args.seed is not None:
        random.seed(args.seed)

    sample = random.choice(dataset)
    *features, label = sample

    features_count = len(features)
    labels_count = len(labels_mapping)

    model_config = dict(
        codebook_size=args.codebook_size,
        features_count=features_count,
        labels_count=labels_count,
        codebook_init_method=args.codebook_init_method,
        codebook_init_dataset=dataset,
    )

    model = lvq.LVQ(**model_config)

    print("Random sample:")
    pprint_vector(sample)

    print("Prediction:", model.predict(features))
    print("Initialized codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Training model...")
    model.train_codebook(
        train_vectors=dataset,
        base_learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        epochs=args.epochs,
    )

    print("Prediction:", model.predict(features))
    print("Trained codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Cross validating model...")
    scores = lvq.cross_validate(
        dataset,
        args.cross_validation_folds,
        **model_config,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        epochs=args.epochs,
    )
    print("Cross validation scores:", [round(score, 3) for score in scores])
    print("Average:", round(sum(scores) / len(scores), 3))
