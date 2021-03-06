import utils
import numpy as np
import argparse

_verbose = False


def main():
    parse_args()
    np.random.seed(10)  # Arbitrary seed for reliable testing
    _scale_factor = 100
    order = 2
    train_data = read('heart_train_csv.csv', cols=list(range(12)), add_bias=True, order=order)
    train_labels = read('heart_train_csv.csv', cols=[13], add_bias=False)
    train_data = train_data / _scale_factor
    train_labels = np.squeeze(train_labels)  # remove extra dim
    theta = np.random.random(train_data.shape[1])
    theta = fit_logestic(train_data, train_labels, theta, 0.000003, 1000)
    test_data = read('heart_test_csv.csv', cols=list(range(12)), add_bias=True, order=order)  # Features with bias
    test_data = test_data / _scale_factor
    test_labels = read('heart_test_csv.csv', add_bias=False, cols=[13])
    test_labels = np.squeeze(test_labels)
    # print(test_data)
    # print(test_labels.shape)
    # print(theta)
    predict = hypo_logestic(test_data, theta)
    #print(predict)
    #print(test_labels)

    threshhold = 0.50
    clamp(predict, threshhold)

    # for i in range(len(predict)):
    #     print('Pred: %s Actual: %s' % (predict[i], test_labels[i]))
    true_pos, true_neg, false_pos, false_neg = Distill(predict, test_labels)
    print(true_pos, true_neg, false_pos, false_neg)
    print('Accuracy: ', Accuracy(true_positive_count=true_pos, true_negative_count=true_neg, false_positive_count=false_pos, false_negative_count=false_neg))
    prec = Precision(true_pos, false_pos)
    print('Precision: ', prec)
    recall = Recall(true_pos, false_neg)
    print('Recall: ', recall)
    print('F1: ', F1(recall, prec))


def clamp(predict, threshhold):
    for i, val in enumerate(predict):
        if val > threshhold:
            predict[i] = 1
        else:
            predict[i] = 0


def fit_logestic(x: np.array, y_train: np.array, theta: np.array, lr, itirations: int, loss_thresh=0.01):
    loss_hist = list()
    for i in range(itirations):
        Y = hypo_logestic(x, theta)
        # logv('Y '+Y.__str__())
        # dJ = (x * (Y - x).transpose()) / x.shape[1]
        dJ = (np.matmul(np.transpose(x), Y - y_train)) / x.shape[0]
        logv('grad ' + dJ.__str__())
        theta = theta - lr * dJ.transpose()
        # loss = [loss -sum(log(Y).*trainY + log(1-Y).*(1-trainY))/length(trainX)];
        loss = -np.sum((y_train * np.log(Y)) + ((1 - np.log(Y)) * (1 - y_train))) / x.shape[0]
        print('Loss ' + loss.__str__())
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


def Distill(predict, test_labels):
    # for loop the reads the twos array
    # this method return true positive, true negative, false positive, false negative
    true = 1
    false = 0
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    for i in range(len(predict)):
        # true positive case
        if (predict[i] == 1 and test_labels[i] == true):
            true_positive_count = true_positive_count + 1

        # true negative case
        if (predict[i] == 0 and test_labels[i] == false):
            true_negative_count = true_negative_count + 1

        # false pos
        if (predict[i] == 1 and test_labels[i] == false):
            false_positive_count = false_positive_count + 1

        # false neg
        if (predict[i] == 0 and test_labels[i] == true):
            false_negative_count = false_negative_count + 1

    return true_positive_count, true_negative_count, false_positive_count, false_negative_count


def Accuracy(true_positive_count, true_negative_count, false_positive_count, false_negative_count):
    accuracy = ((true_positive_count + true_negative_count) / (false_positive_count + false_negative_count + true_negative_count + true_positive_count)) * 100.0

    return accuracy


def Precision(true_positive_count, false_positive_count):
    precision = (true_positive_count / (true_positive_count + false_positive_count)) * 100

    return precision


def Recall(true_positive_count, false_negative_count):
    recall = (true_positive_count / (true_positive_count + false_negative_count)) * 100
    return recall


def F1(recall, precision):
    f1 = 2 * (recall * precision) / (recall + precision)
    return f1


def read(filename: str, add_bias: bool, numeric: bool = True, skip_first=True, cols=None, order=1):
    logv('Reading file %s' % filename)
    data = utils.read_csv_file(filename, skip_first)  # Still python list
    if cols is not None:
        utils.selected_columns(data, cols)
    data = prepare_data(data, add_bias, numeric, order)  # Numpy array
    return data


def prepare_data(data: list[list], add_bias: bool, numeric: bool = True, order=1):
    if numeric:
        for i, row in enumerate(data):
            for j, element in enumerate(row):
                data[i][j] = float(element)
    data = utils.raise_order(data, order)
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
