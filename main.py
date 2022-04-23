from random import choice, seed
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
    labels_mapping, dataset = load_ionosphere_data()
    # labels_mapping, dataset = load_iris_data()
    
    print("Label mapping:")
    for label in labels_mapping:
        print(f"\t{label + ':':20} {labels_mapping[label]}")

    seed(0)

    sample = choice(dataset)
    *features, label = sample

    features_count = len(features)
    labels_count = len(labels_mapping)

    model_config = dict(
        codebook_size=6,
        features_count=features_count,
        labels_count=labels_count,
        codebook_init_method="sample",
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
    model.train_codebook(train_vectors=dataset, base_learning_rate=0.001, epochs=100)

    print("Prediction:", model.predict(features))
    print("Trained codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Cross validating model...")
    scores = lvq.cross_validate(
        dataset,
        5,
        **model_config,
        learning_rate=0.01,
        epochs=100,
    )
    print("Cross validation scores:", [round(score, 3) for score in scores])
    print("Average:", round(sum(scores) / len(scores), 3))
