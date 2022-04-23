import csv
from random import choice

import lvq
from data import normalize


if __name__ == "__main__":
    with open("iris-data.csv", "rt") as f:
        dataset = [[float(value) for value in line] for line in csv.reader(f) if line]
    dataset = normalize(dataset)

    model_config = dict(
        codebook_size=6,
        features_count=4,
        labels_count=3,
        codebook_init_method="sample",
        codebook_init_dataset=dataset,
    )

    model = lvq.LVQ(**model_config)

    sample = choice(dataset)
    print("Random sample:", [round(value, 3) for value in sample])
    *features, label = sample

    print("Prediction:", model.predict(features))
    print("Initialized codebook:")
    for vector in model.codebook:
        print("\t", [round(value, 3) for value in vector])

    print("Training model...")
    model.train_codebook(train_vectors=dataset, base_learning_rate=0.001, epochs=100)

    print("Prediction:", model.predict(features))
    print("Trained codebook:")
    for vector in model.codebook:
        print("\t", [round(value, 3) for value in vector])

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
