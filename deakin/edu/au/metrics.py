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
from treelib import Node, Tree
from prettytable import PrettyTable


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


def get_consistency(y_pred: list, tree: Tree):
    """
    This methods estimates the consistency.

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: np.array
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: value of consistency.
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]
    consistency = []
    for j in range(len(y_pred[0])):
        v = 1
        for i in range(len(y_pred) - 1):
            parent = 'L' + str(i) + '_' + str(y_pred[i][j])
            child = 'L' + str(i + 1) + '_' + str(y_pred[i + 1][j])
            if tree.parent(child).identifier != parent:
                v = 0
                break
        consistency.append(v)
    return np.mean(consistency)


def get_hierarchical_metrics(y_true: list, y_pred: list, tree: Tree):
    """
    This method compute the hierarchical precision/recall/F1-Score. For more details, see:

    Kiritchenko S., Matwin S., Nock R., Famili A.F. (2006) Learning and Evaluation
    in the Presence of Class Hierarchies: Application to Text Categorization. In: Lamontagne L.,
    Marchand M. (eds) Advances in Artificial Intelligence. Canadian AI 2006. Lecture Notes in
    Computer Science, vol 4013. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11766247_34

    :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
    :type y_pred: list
    :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
    :type y_true: list
    :param tree: A tree of the taxonomy.
    :type tree: Tree
    :return: the hierarchical precision/recall/F1-Score values
    :rtype: float
    """
    y_pred = [np.argmax(x, axis=1) for x in y_pred]

    if len(y_true) != len(y_pred):
        raise Exception('Shape of the inputs should be the same')
    hP_list = []
    hR_list = []
    hF1_list = []
    for j in range(len(y_true[0])):
        y_true_aug = set()
        y_pred_aug = set()
        for i in range(len(y_true)):
            true_c = 'L' + str(i) + '_' + str(y_true[i][j][0])
            y_true_aug.add(true_c)
            while tree.parent(true_c) != None:
                true_c = tree.parent(true_c).identifier
                y_true_aug.add(true_c)

            pred_c = 'L' + str(i) + '_' + str(y_pred[i][j])
            y_pred_aug.add(pred_c)
            while tree.parent(pred_c) != None:
                pred_c = tree.parent(pred_c).identifier
                y_pred_aug.add(pred_c)

        y_true_aug.remove('root')
        y_pred_aug.remove('root')

        hP = len(y_true_aug.intersection(y_pred_aug)) / len(y_pred_aug)
        hR = len(y_true_aug.intersection(y_pred_aug)) / len(y_true_aug)
        if 2 * hP + hR != 0:
            hF1 = 2 * hP * hR / (hP + hR)
        else:
            hF1 = 0

        hP_list.append(hP)
        hR_list.append(hR)
        hF1_list.append(hF1)
    return np.mean(hP_list), np.mean(hR_list), np.mean(hF1_list)


def performance_report(y_true: list, y_pred: list, tree: Tree, title=None):
    """
        Build a text report showing the main classification metrics.

        :param y_pred: a 2d array where d1 is the taxonomy level, and d2 is the prediction for each example.
        :type y_pred: list
        :param y_true: a 2d array where d1 is the taxonomy level, and d2 is the ground truth for each example.
        :type y_true: list
        :param tree: A tree of the taxonomy.
        :type tree: Tree
        :param title: A title for the report.
        :type title: str
        :return: the hierarchical precision/recall/F1-Score values
        :rtype: float
        """
    accuracy = get_top_k_taxonomical_accuracy(y_true, y_pred)
    exact_match = get_exact_match(y_true, y_pred)
    consistency = get_consistency(y_pred, tree)
    hP, hR, hF1 = get_hierarchical_metrics(y_true, y_pred, tree)
    t = PrettyTable(['Metric1', 'Value1', 'Metric2', 'Value2', 'Metric3', 'Value3'])
    if title != None:
        t.title = title
    t.add_row(['Exact Match', "{:.4f}".format(exact_match),
               'Consistency', "{:.4f}".format(consistency),
               '-', '-'])
    t.add_row(['h-Precision', "{:.4f}".format(hP),
               'h-Recall', "{:.4f}".format(hR),
               'h-F1-Score', "{:.4f}".format(hF1)])
    row = []
    for i in range(len(accuracy)):
        row.append('Accuracy L_' + str(i))
        row.append("{:.4f}".format(accuracy[i]))
    t.add_row(row)
    t.add_row(['h-Accuracy-k=1', "{:.4f}".format(get_h_accuracy(y_true, y_pred, k=1)),
               'h-Accuracy-k=2', "{:.4f}".format(get_h_accuracy(y_true, y_pred, k=2)),
               'h-Accuracy-k=5', "{:.4f}".format(get_h_accuracy(y_true, y_pred, k=5))])

    t.add_row(['m-Accuracy-k=1', "{:.4f}".format(get_m_accuracy(y_true, y_pred, k=1)),
               'm-Accuracy-k=2', "{:.4f}".format(get_m_accuracy(y_true, y_pred, k=2)),
               'm-Accuracy-k=5', "{:.4f}".format(get_m_accuracy(y_true, y_pred, k=5))])

    print(t)


if __name__ == '__main__':
    y = [[1, 0, 1, 0, 0], [1, 2, 3, 4, 0], [3, 4, 5, 8, 0]]

    # y_pred = [[0, 1, 1, 0, 0], [1, 2, 1, 4, 0], [3, 1, 5, 8, 0]]
    #
    # taxo = [[[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]],
    #         [[1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0],
    #          [0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]
    #         ]
    #
    # print(get_exact_match(y, y_pred))
    # print(get_consistency(y_pred, taxo))
