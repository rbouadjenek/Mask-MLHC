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

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)


class Dataset:

    def __init__(self, name, dataset_path, train_labels_path, test_labels_path, image_size=(64, 64), batch_size=32):
        self.name = name
        self.image_size_ = image_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        # Training set
        train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0)
        train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True)
        self.train_labels_df = train_labels_df
        self.train_dataset = self.get_pipeline(train_labels_df)
        # Splitting into val and test sets
        test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0)
        test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True)
        split = int(len(test_labels_df) * 0.50)
        val_labels_df = test_labels_df[:split]
        self.val_labels_df = val_labels_df
        test_labels_df = test_labels_df[split:]
        self.test_labels_df = test_labels_df
        # Validation set
        self.val_dataset = self.get_pipeline(val_labels_df)
        # Test set
        self.test_dataset = self.get_pipeline(test_labels_df)
        # Number of classes
        self.num_classes_l0 = len(set(train_labels_df['class_level_0']))
        self.num_classes_l1 = len(set(train_labels_df['class_level_1']))
        self.num_classes_l2 = len(set(train_labels_df['class_level_2']))
        self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2]
        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        for (t, c) in zip(list(train_labels_df['class_level_0']), list(train_labels_df['class_level_1'])):
            m0[t][c] = 1
        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(list(train_labels_df['class_level_1']), list(train_labels_df['class_level_2'])):
            m1[t][c] = 1
        self.taxonomy = [m0, m1]
        # Build the labels
        self.labels = []
        labels = ['' for x in range(self.num_classes_l0)]
        for (l, c) in zip(list(train_labels_df['label_level_0']), list(train_labels_df['class_level_0'])):
            labels[c] = l
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l1)]
        for (l, c) in zip(list(train_labels_df['label_level_1']), list(train_labels_df['class_level_1'])):
            labels[c] = l
        self.labels.append(labels)

        labels = ['' for x in range(self.num_classes_l2)]
        for (l, c) in zip(list(train_labels_df['label_level_2']), list(train_labels_df['class_level_2'])):
            labels[c] = l
        self.labels.append(labels)

    def encode_single_sample(self, img_path, class_level_0, class_level_1, class_level_2, fname):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_image(img, expand_animations=False)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, self.image_size_)
        if self.output_level == 'last_level':
            return img, class_level_2
        if self.output_level == 'all':
            return img, (class_level_0, class_level_1, class_level_2, fname)
        else:
            return img, (class_level_0, class_level_1, class_level_2)

    def get_pipeline(self, dataframe, output_level=None):
        self.output_level = output_level
        dataset = tf.data.Dataset.from_tensor_slices(([self.dataset_path + '/' + x for x in dataframe['fname']],
                                                      list(dataframe['class_level_0']),
                                                      list(dataframe['class_level_1']),
                                                      list(dataframe['class_level_2']),
                                                      list(dataframe['fname'])))
        dataset = (
            dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(self.batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

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


def get_Stanford_Cars(image_size=(64, 64), batch_size=32):
    # Get images
    train_data_url = 'http://ai.stanford.edu/~jkrause/car196/car_ims.tgz'
    dataset_path = keras.utils.get_file('car_ims', train_data_url, untar=True)
    # Get labels for training set
    train_labels_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_train_labels.txt'
    train_labels_path = keras.utils.get_file("stanford_cars_train_labels.csv", train_labels_url)
    # Get labels for test set
    test_labels_url = 'https://rbouadjenek.github.io/datasets/stanford_cars_test_labels.txt'
    test_labels_path = keras.utils.get_file("stanford_cars_test_labels.csv", test_labels_url)

    return Dataset('stanford_cars', dataset_path, train_labels_path, test_labels_path, image_size, batch_size)


def get_CU_Birds_200_2011(image_size=(64, 64), batch_size=32):
    # Get images
    train_data_url = 'http://206.12.93.90:8080/CUB_200_2011/CUB_200_2011_v0.2.tar.gz'
    dataset_path = keras.utils.get_file('CUB_200_2011_v0.2', train_data_url, untar=True)
    # Get labels for training set
    train_labels_url = 'https://rbouadjenek.github.io/datasets/cu_birds_train_labels.csv'
    train_labels_path = keras.utils.get_file("cu_birds_train_labels.csv", train_labels_url)
    # Get labels for test set
    test_labels_url = 'https://rbouadjenek.github.io/datasets/cu_birds_test_labels.csv'
    test_labels_path = keras.utils.get_file("cu_birds_test_labels.csv", test_labels_url)

    return Dataset('CU_Birds_200_2011', dataset_path, train_labels_path, test_labels_path, image_size, batch_size)


if __name__ == '__main__':
    dataset = get_Stanford_Cars(image_size=(32, 32))
    print(dataset.num_classes_l0)
    print(dataset.num_classes_l1)
    print(dataset.num_classes_l2)
    print(dataset.taxonomy)
