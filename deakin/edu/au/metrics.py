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


def get_accuracy(y: list, y_pred: list):
    """
    This method computes the accuracy for each level in the taxonomy.

    :param y: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y: list
    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :return: accuracy for each level of the taxonomy.
    :rtype: list
    """
    if len(y) != len(y_pred):
        raise Exception('Size of the inputs should be the same.')
    accuracy = [accuracy_score(y_, y_pred_) for y_, y_pred_ in zip(y, y_pred)]
    return accuracy


def get_exact_match(y: list, y_pred: list):
    """
    This method compute the exact match score. Exact match is defined as the #of examples for
    which the predictions for all level in the taxonomy is correct by the total #of examples.

    :param y: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y: list
    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :return: the exact match value
    :rtype: float
    """
    if len(y) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    exact_match = []
    for j in range(len(y[0])):
        v = 1
        for i in range(len(y)):
            if y[i][j] != y_pred[i][j]:
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

    print(get_accuracy(y, y_pred))
    print(get_exact_match(y, y_pred))
    print(get_consistency(y_pred, taxo))
