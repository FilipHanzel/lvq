import csv
from pprint import pprint

from lvq import LVQ

if __name__ == "__main__":
    with open("iris-data.csv", "rt") as f:
        data = [[float(value) for value in line] for line in csv.reader(f) if line]

    features = 4
    labels = 3
    codebook = 10

    lvq = LVQ(codebook_size=codebook, features_count=features, labels_count=labels)

    print("Sample:", data[0])
    print("Prediction:", lvq.predict(data[0]))

    print("New codebook:")
    pprint(lvq.codebook)

    lvq.train_codebook(train_vectors=data, learning_rate=0.001, epochs=100)

    print("Trained codebook:")
    pprint(lvq.codebook)

    print("Sample:", data[0])
    print("Prediction:", lvq.predict(data[0]))
