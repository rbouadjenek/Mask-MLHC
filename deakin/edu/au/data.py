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
from keras.datasets import cifar100
from graphviz import Digraph
import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
from tensorflow import keras


# mapping_fine_to_cluster = {0: 5, 1: 2, 2: 3, 3: 6, 4: 6, 5: 0, 6: 2, 7: 2, 8: 8, 9: 1, 10: 1, 11: 3, 12: 8,
#                                13: 8, 14: 2, 15: 6, 16: 1, 17: 8, 18: 2, 19: 6, 20: 0, 21: 3, 22: 1, 23: 7, 24: 2,
#                                25: 0, 26: 2, 27: 6, 28: 1, 29: 6, 30: 6, 31: 6, 32: 6, 33: 4, 34: 6, 35: 3, 36: 6,
#                                37: 8, 38: 6, 39: 1, 40: 1, 41: 8, 42: 6, 43: 6, 44: 2, 45: 2, 46: 3, 47: 4, 48: 8,
#                                49: 7, 50: 6, 51: 5, 52: 4, 53: 5, 54: 5, 55: 6, 56: 4, 57: 5, 58: 8, 59: 4, 60: 7,
#                                61: 1, 62: 5, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 68: 7, 69: 1, 70: 5, 71: 7, 72: 6,
#                                73: 2, 74: 6, 75: 6, 76: 1, 77: 2, 78: 2, 79: 2, 80: 6, 81: 8, 82: 2, 83: 5, 84: 0,
#                                85: 8, 86: 1, 87: 0, 88: 6, 89: 8, 90: 8, 91: 2, 92: 5, 93: 6, 94: 0, 95: 6, 96: 4,
#                                97: 6, 98: 3, 99: 2}


class Cifar100:
    cifar100_mapping_coarse_to_top = {0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 0, 12: 0,
                                      13: 0, 14: 0,
                                      15: 0, 16: 0, 17: 0, 18: 1, 19: 1}

    labels = [['bio organism', 'objects'],
              ['aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetales',
               'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
               'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herivores',
               'medium_mammals', 'non-insect_inverterates', 'people', 'reptiles', 'small_mammals', 'trees',
               'vehicles_1', 'vehicles_2'],
              ['apple', 'aquarium_fish', 'ray', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'owl',
               'boy',
               'ridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
               'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
               'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyoard', 'lamp', 'lawn_mower',
               'leopard', 'lion', 'lizard', 'loster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
               'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rait', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
               'streetcar', 'sunflower', 'sweet_pepper', 'tale', 'tank', 'telephone', 'television', 'tiger', 'tractor',
               'train', 'trout', 'tulip', 'turtle', 'wardroe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']]

    def __init__(self):
        """

        :param type: to indicate if to use coarse classes as given in the cifar100 dataset or use clusters.
        :type type: str
        """
        (X_c_train, y_c_train), (X_c_test, y_c_test) = cifar100.load_data(label_mode='coarse')
        (X_f_train, y_f_train), (X_f_test, y_f_test) = cifar100.load_data(label_mode='fine')

        y_top_train = self.map_fine_to_cluster_cifar100(y_c_train, self.cifar100_mapping_coarse_to_top)
        y_top_test = self.map_fine_to_cluster_cifar100(y_c_test, self.cifar100_mapping_coarse_to_top)

        self.X_train = X_f_train
        self.X_val = X_f_test[:5000]
        self.X_test = X_f_test[5000:]

        self.y_train = [y_top_train, y_c_train, y_f_train]
        self.y_val = [y_top_test[:5000], y_c_test[:5000], y_f_test[:5000]]
        self.y_test = [y_top_test[5000:], y_c_test[5000:], y_f_test[5000:]]

        self.image_size = self.X_train[0].shape

        self.num_classes_l0 = len(set([v[0] for v in y_top_train]))
        self.num_classes_l1 = len(set([v[0] for v in y_c_train]))
        self.num_classes_l2 = len(set([v[0] for v in y_f_train]))

        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        for (t, c) in zip(y_top_train, y_c_train):
            t = t[0]
            c = c[0]
            m0[t][c] = 1

        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(y_c_train, y_f_train):
            t = t[0]
            c = c[0]
            m1[t][c] = 1
        self.taxonomy = [m0, m1]

    def map_fine_to_cluster_cifar100(self, y, mapping):
        """
        This function is only used to create label for clusters if used.  Clusters are obtained from:

        :param y:
        :type y:
        :return:
        :rtype:
        """
        # Mapping fine -> cluster

        y_top = []
        for f in y:
            k = f[0]
            c = np.array([mapping[k]])
            y_top.append(c)
        return np.array(y_top)

    def draw_taxonomy(self):
        return draw_taxonomy(self.taxonomy, self.labels)


def load_dataset(labels_path, images_path, image_size):
    df = pd.read_csv(labels_path)
    filenames = df['fname'].values
    label_level_0 = df['label_level_0'].values
    class_level_0 = np.array([[x] for x in df['class_level_0'].values])
    label_level_1 = df['label_level_1'].values
    class_level_1 = np.array([[x] for x in df['class_level_1'].values])
    label_level_2 = df['label_level_2'].values
    class_level_2 = np.array([[x] for x in df['class_level_2'].values])

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        images_path,
        label_mode=None,
        color_mode="rgb",
        batch_size=1,
        image_size=image_size,
        shuffle=False,
    )

    dataset_list = [[] for x in range(filenames.size)]
    for img, fname1 in zip(dataset, dataset.file_paths):
        for i, fname2 in enumerate(filenames):
            if fname2 in fname1:
                dataset_list[i] = list(img[0])
                break
    dataset_np = np.stack(dataset_list)
    dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2 = shuffle(
        dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2,
        random_state=0)
    return dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2


class Stanford_Cars:

    def __init__(self, image_size):
        self.image_size = image_size
        train_data_url = 'http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
        filename = 'car_ims'
        print('Preparing dataset..')
        dataset_path = keras.utils.get_file(filename, train_data_url, untar=True)
        train_csv_url = 'https://rbouadjenek.github.io/datasets/stanford_car_train_labels.txt'
        train_label_path = keras.utils.get_file("stanford_car_train_labels.csv", train_csv_url)
        test_csv_url = 'https://rbouadjenek.github.io/datasets/stanford_car_test_labels.txt'
        test_label_path = keras.utils.get_file("stanford_car_test_labels.csv", test_csv_url)
        X_train, class_train_level_0, y_train_level_0, class_train_level_1, y_train_level_1, class_train_level_2, y_train_level_2 = load_dataset(
            labels_path=train_label_path, images_path=dataset_path,
            image_size=self.image_size)
        X_test, class_test_level_0, y_test_level_0, class_test_level_1, y_test_level_1, class_test_level_2, y_test_level_2 = load_dataset(
            labels_path=test_label_path,
            images_path=dataset_path,
            image_size=self.image_size)

        self.X_train = X_train
        self.X_val = X_test[:2500]
        self.X_test = X_test[2500:]

        self.y_train = [y_train_level_0, y_train_level_1, y_train_level_2]
        self.y_val = [y_test_level_0[:2500], y_test_level_1[:2500], y_test_level_2[:2500]]
        self.y_test = [y_test_level_0[2500:], y_test_level_1[2500:], y_test_level_2[2500:]]

        self.num_classes_l0 = len(set([v[0] for v in y_train_level_0]))
        self.num_classes_l1 = len(set([v[0] for v in y_train_level_1]))
        self.num_classes_l2 = len(set([v[0] for v in y_train_level_2]))

        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        for (t, c) in zip(y_train_level_0, y_train_level_1):
            t = t[0]
            c = c[0]
            m0[t][c] = 1

        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(y_train_level_1, y_train_level_2):
            t = t[0]
            c = c[0]
            m1[t][c] = 1
        self.taxonomy = [m0, m1]

        # Build the labels
        self.labels = []
        labels = ['' for x in range(self.num_classes_l0)]
        for i, idx in enumerate(y_train_level_0):
            labels[idx[0]] = class_train_level_0[i]
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l1)]
        for i, idx in enumerate(y_train_level_1):
            labels[idx[0]] = class_train_level_1[i]
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l2)]
        for i, idx in enumerate(y_train_level_2):
            labels[idx[0]] = class_train_level_2[i]
        self.labels.append(labels)

    def draw_taxonomy(self):
        return draw_taxonomy(self.taxonomy, self.labels)


def draw_taxonomy(taxonomy, LABELS):
    """
    This method draws the taxonomy using the graphviz library.
    :return:
    :rtype: Digraph
     """
    u = Digraph('unix', node_attr={'color': 'lightblue2', 'style': 'filled'}, strict=True)
    u.attr(size='6,6')
    u.attr(rankdir="LR")

    for i in range(len(taxonomy[0])):
        u.edge('root', LABELS[0][i], LABELS[0][i])

    for l in range(len(taxonomy)):
        for i in range(len(taxonomy[l])):
            for j in range(len(taxonomy[l][i])):
                if taxonomy[l][i][j] == 1:
                    u.edge(LABELS[l][i], LABELS[l + 1][j])

    return u


if __name__ == '__main__':
    dataset = Stanford_Cars(image_size=(32, 32))
    print(dataset.num_classes_l0)
    print(dataset.num_classes_l1)
    print(dataset.num_classes_l2)
    print(dataset.taxonomy)
