import csv
from pprint import pprint

import lvq
from data import normalize

if __name__ == "__main__":
    with open("iris-data.csv", "rt") as f:
        dataset = [[float(value) for value in line] for line in csv.reader(f) if line]
    dataset = normalize(dataset)

    features = 4
    labels = 3
    codebook = 10

    model = lvq.LVQ(
        codebook_size=codebook, features_count=features, labels_count=labels
    )

    print("Sample:", dataset[0])
    print("Prediction:", model.predict(dataset[0]))

    print("New codebook:")
    pprint(model.codebook)

    model.train_codebook(train_vectors=dataset, base_learning_rate=0.001, epochs=100)

    print("Trained codebook:")
    pprint(model.codebook)

    print("Sample:", dataset[0])
    print("Prediction:", model.predict(dataset[0]))

    scores = lvq.cross_validate(
        dataset,
        5,
        codebook_size=codebook,
        features_count=features,
        labels_count=labels,
        learning_rate=0.01,
        epochs=100,
    )
    print("Cross validation scores:")
    pprint(scores)
