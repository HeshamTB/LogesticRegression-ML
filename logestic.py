import math

import utils
import numpy as np
import argparse
import numba
from numba import jit


_verbose = False


def main():
    parse_args()
    np.random.seed(10)  # Arbitrary seed for reliable testing
    _scale_factor = 100
    # Prepare data
    train_data = read('heart_train_csv.csv', cols=list(range(12)), add_bias=True)
    train_labels = read('heart_train_csv.csv', cols=[13], add_bias=False)
    train_data = train_data / _scale_factor
    train_labels = np.squeeze(train_labels)  # remove extra dim
    theta = np.random.random(train_data.shape[1])
    # Fit (learn)
    theta = fit_logestic(train_data, train_labels, theta, 0.3, 100)
    # Evaluate metrics
    test_data = read('heart_test_csv.csv', cols=list(range(12)), add_bias=True)  # Features with bias
    test_data = test_data / _scale_factor
    test_labels = read('heart_test_csv.csv', add_bias=False, cols=[13])
    test_labels = np.squeeze(test_labels)
    predict = hypo_logestic(test_data, theta)
    print(predict)
    # TODO: Find acc, prec, recall, F1
    for i, val in enumerate(predict):
        if val > 0.5:
            print('1 ', end='')
        else:
            print('0 ', end='')
        print(int(test_labels[i]))


@jit(nopython=True)
def fit_logestic(x: np.array, y_train: np.array, theta: np.array, lr, itirations: int, loss_thresh=0.01):
    #loss_hist = list()
    for i in range(itirations):
        #Y = hypo_logestic(x, theta)
        #Y = np.matmul(x, theta)
        Y = np.zeros_like(theta)
        for i in range(x):
            for j in range(theta[0]):
                for k in range(theta):
                    Y[i][j] += x[i][k] * theta[k][j]

        Y = 1 / (1 + np.exp(-Y))
        # logv('Y '+Y.__str__())
        # dJ = (x * (Y - x).transpose()) / x.shape[1]
        dJ = (np.matmul(np.transpose(x), Y - y_train)) / x.shape[0]
        #logv('grad ' + dJ.__str__())
        theta = theta - lr * dJ.transpose()
        # loss = [loss -sum(log(Y).*trainY + log(1-Y).*(1-trainY))/length(trainX)];
        loss = -np.sum(y_train * np.log(Y) + (1 - np.log(Y)) * (1 - y_train)) / x.shape[0]
        #print('Loss ' + loss.__str__())
        # Break conditions
    return theta


def hypo_logestic(x: np.array, theta: np.array) -> np.array:
    # Hot path. Maybe inline this to reduce func call overhead.
    logv('x ' + x.__str__())
    logv('theta ' + theta.transpose().__str__())
    Y = np.matmul(x, theta)
    logv('Y ' + Y.__str__())
    # print('Y ',Y.shape)
    Y = 1 / (1 + np.exp(-Y))
    logv('Y sigmoid ' + Y.__str__())
    # print('Y ', Y.shape)
    return Y


def read(filename: str, add_bias: bool, numeric: bool = True, skip_first=True, cols=None):
    logv('Reading file %s' % filename)
    data = utils.read_csv_file(filename, skip_first)  # Still python list
    if cols is not None:
        utils.selected_columns(data, cols)
    data = prepare_data(data, add_bias, numeric)  # Numpy array
    return data


def prepare_data(data: list[list], add_bias: bool, numeric: bool = True):
    if numeric:
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                data[i][j] = float(element)
    if add_bias:
        # With this we double itirate on the same data when numeric and add bias.
        for i, row in enumerate(data):
            row.insert(0, 1)  # insert at top of list
    data_mat = np.array(data)
    return data_mat


def logv(msg: str):
    global _verbose
    if _verbose:
        print(msg)
    # Add levels?


def parse_args():
    global _verbose
    parser = argparse.ArgumentParser(prog="logestic")
    parser.add_argument("-v", '--verbose', help='be verbose', action='store_true')
    varss = vars(parser.parse_args())
    if varss['verbose']: _verbose = True


if __name__ == '__main__':
    exit(main())
