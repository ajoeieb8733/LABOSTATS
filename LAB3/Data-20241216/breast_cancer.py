"""Breast cancer dataset helper class

https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"""

from sklearn.datasets import load_breast_cancer
import numpy as np


class BreastCancerDataset:

    def __init__(self, random_state: int = 0):
        self._dataset = load_breast_cancer()
        self._X = self._dataset['data']
        self._Y = self._dataset['target']
        self._columns = self._dataset['feature_names']
        self.n = self._X.shape[0]
        self.train_n = int(0.8*self.n)
        self.test_n = self.n-self.train_n
        np.random.seed(random_state)
        idxs = np.arange(self.n)
        np.random.shuffle(idxs)
        self.train_idxs = idxs[:self.train_n]
        self.test_idxs = idxs[self.train_n:]

    @property
    def Xtrain(self):
        return self._X[self.train_idxs]

    @property
    def Xtest(self):
        return self._X[self.test_idxs]

    @property
    def Ytrain(self):
        return self._Y[self.train_idxs]

    @property
    def Ytest(self):
        return self._Y[self.test_idxs]

    @property
    def columns(self):
        return self._columns



if __name__ == "__main__":
    data = BreastCancerDataset()

    print(f"{data.Xtrain.shape[0]} cases in training set.")
    print(f"{data.Xtest.shape[0]} cases in test set.")

    # Show mean and standard deviation of the training set for each feature
    features = data.columns
    for i in range(len(data.columns)):
        print(features[i], data.Xtrain[:, i].mean(), data.Xtrain[:, i].std(ddof=1))

    # Equivalent:
    # for name, values in zip(data.columns, data.Xtrain.T):
    #     print(name, values.mean(), values.std(ddof=1)) 
