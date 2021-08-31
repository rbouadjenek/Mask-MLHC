# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import hmean
from sklearn.metrics import top_k_accuracy_score

def one_hot(y):
    n_values = np.max(y) + 1
    y_new = np.eye(n_values)[y[:]]
    return y_new


def get_top_k_accuracy_score(y_true: list, y_pred: list, k=1):
    if len(list(y_pred[0])) == 2:
        if k == 1:
            return accuracy_score(y_true, np.argmax(y_pred, axis=1))
        else:
            return 1
    else:
        return top_k_accuracy_score(y_true, y_pred, k=k)


def get_top_k_taxonomical_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the top k accuracy for each level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    if len(y_true) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    accuracy = [get_top_k_accuracy_score(y_, y_pred_, k) for y_, y_pred_ in zip(y_true, y_pred)]
    return accuracy


def get_h_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the harmonic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return hmean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_m_accuracy(y_true: list, y_pred: list, k=1):
    """
    This method computes the harmonic mean of accuracies of all level in the taxonomy.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    return np.mean(get_top_k_taxonomical_accuracy(y_true, y_pred, k))


def get_exact_match(y_true: list, y_pred: list):
    """
    This method compute the exact match score. Exact match is defined as the #of examples for
    which the predictions for all level in the taxonomy is correct by the total #of examples.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :return: the exact match value
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    exact_match = []
    for j in range(len(y_true[0])):
        v = 1
        for i in range(len(y_true)):
            if y_true[i][j] != y_pred[i][j]:
                v = 0
                break
        exact_match.append(v)
    return np.mean(exact_match)


def get_consistency(y_pred: list, taxo: list):
    """
    This methods estimates the consistency.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: np.array
    :param taxo: a
    :type taxo: np.array
    :return: value of consistency.
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    if len(y_pred) - 1 != len(taxo):
        raise Exception('The predictions do not match the taxonomy.')
    consistency = []
    for j in range(len(y_pred[0])):
        v = 1
        for i in range(len(y_pred) - 1):
            l = int(y_pred[i][j])
            l_next = int(y_pred[i + 1][j])
            if taxo[i][l][l_next] == 0:
                v = 0
                break
        consistency.append(v)
    return np.mean(consistency)


if __name__ == '__main__':
    y = [[1, 0, 1, 0, 0], [1, 2, 3, 4, 0], [3, 4, 5, 8, 0]]

    y_pred = [[0, 1, 1, 0, 0], [1, 2, 1, 4, 0], [3, 1, 5, 8, 0]]

    taxo = [[[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]],
            [[1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]
            ]

    print(get_taxonomical_accuracy(y, y_pred))
    print(get_exact_match(y, y_pred))
    print(get_consistency(y_pred, taxo))
