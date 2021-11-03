import utils
import numpy as np


def main():
    train_data = read('heart_train_csv.csv')
    theta = np.random.random(train_data.shape[1])
    theta = np.matrix(theta)
    theta = fit_logestic(train_data, theta.transpose(), 0.00001, 1)



def fit_logestic(x: np.matrix, theta:np.matrix, lr, itirations: int, loss_thresh=0.01):
    Y = hypo_logestic(x, theta)
    dJ = (x * (Y - x).transpose()) / x.shape[1]
    theta = theta - lr*dJ.transpose()
    # Loss
    # Break conditions
    return theta


def hypo_logestic(x: np.matrix, theta: np.matrix):
    # Hot path. Maybe inline this to reduce func call overhead.
    print(x)
    print(theta)
    Y = x * theta
    print(Y)
    Y = 1 / (1 + np.matrix(np.exp(-Y))) # TODO: Not correct
    print(Y)
    return Y


def read(filename: str, numeric: bool = True, add_bias=True, skip_first=True):
    data = utils.read_csv_file(filename, skip_first)
    data = prepare_data(data, numeric, add_bias)
    return data


def prepare_data(data: list[list], add_bias: bool = False, numeric: bool = True):
    if numeric:
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                data[i][j] = float(element)
    if add_bias:
        # With this we double itirate on the same data when numeric and add bias.
        for i, row in enumerate(data):
            row.insert(0, 1)  # insert at top of list
    data_mat = np.matrix(data)
    return data_mat


if __name__ == '__main__':
    exit(main())
