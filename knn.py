import numpy as np
import random
from time import time as time


def euclidianDistance(x_test, x_train):
    return np.linalg.norm(np.subtract(x_train, x_test), axis=1)


def readCSV(trainFilePath="./datasets/train.csv", testFilePath="./datasets/test_pub.csv"):

    # read csv
    # xtrain  =  r x f matrix: r = row and f = features 8000x 85
    dataset = np.genfromtxt(trainFilePath, delimiter=",", skip_header=1)
    x_train = np.genfromtxt(trainFilePath, delimiter=",", skip_header=1, usecols=range(1, 86))
    y_train = np.genfromtxt(trainFilePath, delimiter=",", skip_header=1, usecols=-1)

    # x_test = r x f matrix  2000 x 85
    x_test = np.genfromtxt(testFilePath, delimiter=",", skip_header=1, usecols=range(1, 86))
    return dataset, x_train, y_train, x_test
    # return dataset


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test):
        # store the predictions
        y_pred = np.array([])
        for x in x_test:
            # calculate the euclidian disntance between the test point and all the points in the training set
            distance = euclidianDistance(x, self.x_train)

            # get k nearst points (we can sort this by ditance and return by nearest neighbor)
            neighbors = np.argsort(distance)[: self.k]

            label = {}
            # get the majority class
            for neighbor in neighbors:
                if self.y_train[neighbor] in label:
                    label[self.y_train[neighbor]] += 1
                else:
                    label[self.y_train[neighbor]] = 1
            # return the majority class of the neighbors as prediction
            y_pred = np.append(y_pred, max(label, key=label.get))
        return y_pred

    def accuracy(self, y_val, y_pred):
        return np.mean(y_pred == y_val)


class KFCV:
    def __init__(self, k):
        self.k = k

    def trainTestSplit(self, dataset):
        np.random.shuffle(dataset)
        kSplit = np.array_split(dataset, self.k)
        return kSplit

    def evaluate(self, dataset, classifier):
        # split data set
        train_test_split = self.trainTestSplit(dataset)

        metric = []
        clf = classifier

        print("Evaluating...")
        for i in range(len(train_test_split)):
            #     tic = time()
            trainSet = train_test_split
            testSet = trainSet[i]
            trainSet = np.delete(trainSet, i, axis=0)

            # training set
            x_train = trainSet[:, :, 1:-1]
            y_train = trainSet[:, :, -1:]

            # test set
            x_test = testSet[:, 1:-1]
            y_test = testSet[:, -1:]

            # flatten arrays making sure that they are in the right shape
            x_train = np.concatenate(x_train)
            y_test = np.concatenate(y_test)
            y_train = np.concatenate(np.concatenate(y_train))

            # fit and make predictions
            clf.fit(x_train, y_train)
            predict = clf.predict(x_test)
            accuracy = clf.accuracy(y_test, predict)
            metric.append(accuracy)

            # return
            print("Set ", i + 1, "training accuracy: {:.3f}".format(accuracy))
        mean_accuracy = sum(metric) / len(metric)
        print("validatoin accuracy: {:.3f}".format(mean_accuracy))
        print("variance: ", np.var(metric))
        print()
        return mean_accuracy


if __name__ == "__main__":
    #
    _k = [1, 3, 5, 7, 9, 99, 999, 8000]
    dataset, x_train, y_train, x_test = readCSV()
    # for k in _k:
    # print("k: ", k)
    clf = KNN(89)
    # training accuracy
    print("Evaluating...")
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_train)
    accuracy = clf.accuracy(prediction, y_train)
    print("training accuracy: ", accuracy)

    kFold = KFCV(4)
    result = kFold.evaluate(dataset, clf)

    x_train = x_train[:, :-1]
    x_test = x_test[:, :-1]

    print("Predicting...")
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print("Writing to File...")
    with open("prediction.csv", "w") as f:
        f.write("id,income\n")
        for i, prediction in enumerate(predict):
            f.write("%d,%d\n" % (i, prediction))
    f.close()
    print("Done")