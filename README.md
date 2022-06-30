# LVQ classification algorithm implementation with pure python

*Tested with python 3.8.5*

**LVQ** or **Learning Vector Quantization** is an algorithm similar to KNN, but instead of using the whole dataset it uses smaller, trained codebook.

Three different initialization methods are supported:
- *random* - initialize all weights with random number between 0 and 1 (inclusive)
- *zeros* - initialize all weights with zeros
- *dataset sample* - pick a subset of vectors in training dataset

If LVQ will be initialized with *dataset sample* method and codebook size equal to size of the dataset, we will get KNN, where k=1.

### Datasets
Two datasets are included:
- *iris* - https://archive.ics.uci.edu/ml/datasets/iris
- *ionosphere* - https://archive.ics.uci.edu/ml/datasets/ionosphere

### To be implemented:
- learning rate decay customization
