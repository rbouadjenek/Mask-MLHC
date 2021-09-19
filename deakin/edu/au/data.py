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
from treelib import Tree


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
              ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
               'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
               'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin',
               'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
               'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
               'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
               'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
               'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
               'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
               'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
               'woman', 'worm']]

    def __init__(self):
        """
        :param type: to indicate if to use coarse classes as given in the cifar100 dataset or use clusters.
        :type type: str
        """
        self.name = 'cifar100'
        (X_c_train, y_c_train), (X_c_test, y_c_test) = cifar100.load_data(label_mode='coarse')
        (X_f_train, y_f_train), (X_f_test, y_f_test) = cifar100.load_data(label_mode='fine')

        y_top_train = self.map_fine_to_cluster_cifar100(y_c_train, self.cifar100_mapping_coarse_to_top)
        y_top_test = self.map_fine_to_cluster_cifar100(y_c_test, self.cifar100_mapping_coarse_to_top)

        self.X_train = X_f_train
        self.X_val = X_f_test[:5000]
        self.X_test = X_f_test[5000:]

        self.train_filenames = [str(x) for x in range(len(self.X_train))]
        self.val_filenames = [str(x) for x in range(len(self.X_val))]
        self.test_filenames = [str(x) for x in range(len(self.X_test))]

        self.y_train = [y_top_train, y_c_train, y_f_train]
        self.y_val = [y_top_test[:5000], y_c_test[:5000], y_f_test[:5000]]
        self.y_test = [y_top_test[5000:], y_c_test[5000:], y_f_test[5000:]]

        self.image_size = self.X_train[0].shape

        self.num_classes_l0 = len(set([v[0] for v in y_top_train]))
        self.num_classes_l1 = len(set([v[0] for v in y_c_train]))
        self.num_classes_l2 = len(set([v[0] for v in y_f_train]))
        self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2]
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

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)


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
    dataset_np = dataset_np.astype(int)
    dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2, filenames = shuffle(
        dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2, filenames,
        random_state=0)
    return dataset_np, label_level_0, class_level_0, label_level_1, class_level_1, label_level_2, class_level_2, filenames


class Stanford_Cars:

    def __init__(self, image_size):
        self.name = 'stanford_cars'
        train_data_url = 'http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
        filename = 'car_ims'
        print('Preparing dataset...')
        dataset_path = keras.utils.get_file(filename, train_data_url, untar=True)
        train_csv_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_train_labels.txt'
        train_label_path = keras.utils.get_file("stanford_cars_train_labels.csv", train_csv_url)
        test_csv_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_test_labels.txt'
        test_label_path = keras.utils.get_file("stanford_cars_test_labels.csv", test_csv_url)
        X_train, class_train_level_0, y_train_level_0, class_train_level_1, y_train_level_1, class_train_level_2, y_train_level_2, train_filenames = load_dataset(
            labels_path=train_label_path, images_path=dataset_path,
            image_size=image_size)
        X_test, class_test_level_0, y_test_level_0, class_test_level_1, y_test_level_1, class_test_level_2, y_test_level_2, test_filenames = load_dataset(
            labels_path=test_label_path,
            images_path=dataset_path,
            image_size=image_size)

        self.train_filenames = train_filenames
        self.val_filenames = test_filenames[:4020]
        self.test_filenames = test_filenames[4020:]
        self.X_train = X_train
        self.X_val = X_test[:4020]
        self.X_test = X_test[4020:]

        self.y_train = [y_train_level_0, y_train_level_1, y_train_level_2]
        self.y_val = [y_test_level_0[:4020], y_test_level_1[:4020], y_test_level_2[:4020]]
        self.y_test = [y_test_level_0[4020:], y_test_level_1[4020:], y_test_level_2[4020:]]

        self.num_classes_l0 = len(set([v[0] for v in y_train_level_0]))
        self.num_classes_l1 = len(set([v[0] for v in y_train_level_1]))
        self.num_classes_l2 = len(set([v[0] for v in y_train_level_2]))
        self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2]

        self.image_size = self.X_train[0].shape
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

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)


class CU_Birds_200_2011:

    def __init__(self, image_size):
        self.name = 'CU_Birds_200_2011'
        train_data_url = 'http://206.12.93.90:8080/CUB_200_2011/CUB_200_2011.tgz'
        filename = 'CUB_200_2011'
        print('Preparing dataset...')
        dataset_path = keras.utils.get_file(filename, train_data_url, untar=True)
        train_csv_url = 'https://rbouadjenek.github.io/datasets/cu_birds_train_labels.csv'
        train_label_path = keras.utils.get_file("cu_birds_train_labels.csv", train_csv_url)
        test_csv_url = 'https://rbouadjenek.github.io/datasets/cu_birds_test_labels.csv'
        test_label_path = keras.utils.get_file("cu_birds_test_labels.csv", test_csv_url)
        X_train, class_train_level_0, y_train_level_0, class_train_level_1, y_train_level_1, class_train_level_2, y_train_level_2, train_filenames = load_dataset(
            labels_path=train_label_path, images_path=dataset_path,
            image_size=image_size)
        X_test, class_test_level_0, y_test_level_0, class_test_level_1, y_test_level_1, class_test_level_2, y_test_level_2, test_filenames = load_dataset(
            labels_path=test_label_path,
            images_path=dataset_path,
            image_size=image_size)

        self.train_filenames = train_filenames
        self.val_filenames = test_filenames[:3000]
        self.test_filenames = test_filenames[3000:]
        self.X_train = X_train
        self.X_val = X_test[:3000]
        self.X_test = X_test[3000:]

        self.y_train = [y_train_level_0, y_train_level_1, y_train_level_2]
        self.y_val = [y_test_level_0[:3000], y_test_level_1[:3000], y_test_level_2[:3000]]
        self.y_test = [y_test_level_0[3000:], y_test_level_1[3000:], y_test_level_2[3000:]]

        self.num_classes_l0 = len(set([v[0] for v in y_train_level_0]))
        self.num_classes_l1 = len(set([v[0] for v in y_train_level_1]))
        self.num_classes_l2 = len(set([v[0] for v in y_train_level_2]))
        self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2]

        self.image_size = self.X_train[0].shape
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

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)


def get_tree(taxonomy, labels):
    """
    This method draws the taxonomy using the graphviz library.
    :return:
    :rtype: Digraph
     """
    tree = Tree()
    tree.create_node("Root", "root")  # root node

    for i in range(len(taxonomy[0])):
        tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

    for l in range(len(taxonomy)):
        for i in range(len(taxonomy[l])):
            for j in range(len(taxonomy[l][i])):
                if taxonomy[l][i][j] == 1:
                    tree.create_node(labels[l + 1][j] + ' -> (L' + str(l + 1) + '_' + str(j) + ')',
                                     'L' + str(l + 1) + '_' + str(j),
                                     parent='L' + str(l) + '_' + str(i))

    return tree


if __name__ == '__main__':
    dataset = CU_Birds_200_2011(image_size=(32, 32))
    print(dataset.num_classes_l0)
    print(dataset.num_classes_l1)
    print(dataset.num_classes_l2)
    print(dataset.taxonomy)
