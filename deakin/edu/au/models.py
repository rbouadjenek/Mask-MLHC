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
from tensorflow import keras
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Activation, Lambda, Conv2D, MaxPool2D, \
    GlobalAveragePooling2D, Multiply, Concatenate, experimental
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.applications import VGG19, VGG16, ResNet50, Xception
from deakin.edu.au.data import Cifar100
import deakin.edu.au.metrics as metrics
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB0
from prettytable import PrettyTable


class performance_callback(keras.callbacks.Callback):
    def __init__(self, X, y, tree, name=None):
        self.X = X
        self.y = y
        self.tree = tree
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        metrics.performance_report(self.y, y_pred, self.tree, title=self.name)


def get_mout_model(num_classes: list,
                   image_size,
                   conv_base='vgg19',
                   learning_rate=1e-5,
                   loss_weights=[],
                   lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))

    # Build the model
    model = Model(name='mout_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_BCNN1(num_classes: list,
              image_size,
              reverse=False,
              conv_base='vgg19',
              learning_rate=1e-5,
              loss_weights=[],
              lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    if reverse:
        num_classes = list(reversed(num_classes))
    for idx, v in enumerate(num_classes):
        if reverse:
            idx = len(num_classes) - idx - 1
        if len(out_layers) == 0:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))
        else:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(out_layers[-1]))
    if reverse:
        out_layers = list(reversed(out_layers))
    # Build the model
    model = Model(name='Model_BCNN1_reversed_' + str(reverse),
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_BCNN2(num_classes: list,
              image_size,
              reverse=False,
              conv_base='vgg19',
              learning_rate=1e-5,
              loss_weights=[],
              lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    logits_layers = []
    out_layers = []
    if reverse:
        num_classes = list(reversed(num_classes))
    for idx, v in enumerate(num_classes):
        if reverse:
            idx = len(num_classes) - idx - 1
        if len(logits_layers) == 0:
            logits = Dense(v, name='logits_level_' + str(idx))(conv_base)
            out_layers.append(Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(Activation(keras.activations.relu)(logits))
        else:
            logits = Dense(v, name='logits_level_' + str(idx))(logits)
            out_layers.append(Activation(keras.activations.softmax, name='out_level_' + str(idx))(logits))
            logits_layers.append(Activation(keras.activations.relu)(logits))

    if reverse:
        out_layers = list(reversed(out_layers))
    # Build the model
    model = Model(name='Model_BCNN2_reversed_' + str(reverse),
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


def get_mnets(num_classes: list,
              image_size,
              conv_base='vgg19',
              learning_rate=1e-5,
              loss_weights=[],
              lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base_list = [get_conv_base(conv_base, regularizer=regularizer) for x in num_classes]
    out_layers = []
    for i in range(len(conv_base_list)):
        conv_base_list[i]._name = 'conv_base' + str(i)
        conv_base_list[i] = conv_base_list[i](in_layer)
        conv_base_list[i] = Flatten()(conv_base_list[i])
        out_layers.append(Dense(num_classes[i], activation="softmax", name='out_level_' + str(i))(conv_base_list[i]))

    # Build the model
    model = Model(name='mnets',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


class BaselineModel(Model):
    def __init__(self, taxonomy, *args, **kwargs):
        super(BaselineModel, self).__init__(*args, **kwargs)
        self.taxonomy = taxonomy

    def predict(self, X):
        pred = super().predict(X)
        out = []
        for i in range(len(self.taxonomy) + 1):
            out.append([])
        for v in pred:
            child = np.argmax(v)
            out[-1].append(v)
            for i in reversed(range(len(self.taxonomy))):
                m = self.taxonomy[i]
                row = list(np.transpose(m)[child])
                parent = row.index(1)
                child = parent
                one_hot = np.zeros(len(row))
                one_hot[child] = 1
                # one_hot = np.random.uniform(low=1e-6, high=1e-5, size=len(row))
                # one_hot[child] = 1 - (np.sum(one_hot) - one_hot[child])
                out[i].append(one_hot)
        return out


# def get_Baseline_model(num_classes: list, image_size, taxonomy, conv_base=VGG19(include_top=False, weights="imagenet"),
#                        learning_rate=1e-5):
#     # Conv base
#     in_layer = Input(shape=image_size, name='main_input')
#     conv_base = conv_base(in_layer)
#     conv_base = Flatten()(conv_base)
#     # create output layers
#     out_layers = []
#     for idx, v in enumerate(num_classes):
#         out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))
#
#     # Build the model
#     model = BaselineModel(taxonomy=taxonomy, name='baseline_model',
#                           inputs=in_layer,
#                           outputs=out_layers)
#     loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
#     loss_weights = [1 for x in num_classes]
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer,
#                   loss=loss,
#                   loss_weights=loss_weights,
#                   metrics=['accuracy'])
#     return model

def get_Baseline_model(num_classes: list,
                       image_size,
                       taxonomy, conv_base='vgg19',
                       learning_rate=1e-5,
                       lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layer = Dense(num_classes[-1], activation="softmax", name='output')(conv_base)
    # Build the model
    model = BaselineModel(taxonomy=taxonomy, name='baseline_model',
                          inputs=in_layer,
                          outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_Classifier_model(num_classes,
                         image_size,
                         conv_base='vgg19',
                         learning_rate=1e-5,
                         lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size)
    # rescale = experimental.preprocessing.Rescaling(1. / 255)(in_layer)
    conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layer = Dense(num_classes, kernel_regularizer=regularizer, activation="softmax")(conv_base)
    # Build the model
    model = Model(inputs=in_layer,
                  outputs=out_layer)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    return model


def get_MLPH_model(num_classes: list,
                   image_size,
                   learning_rate=1e-5,
                   loss_weights=[],
                   lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base('vgg19', regularizer=regularizer)
    # conv_base.summary()
    layer_outputs = [conv_base.get_layer("block5_conv1").output,
                     conv_base.get_layer("block5_conv2").output,
                     conv_base.get_layer("block5_conv3").output]
    conv_base_model = Model(inputs=conv_base.input, outputs=layer_outputs)
    conv_base = conv_base_model(in_layer)
    relu5_1_X = Flatten()(conv_base[0])
    relu5_2_Y = Flatten()(conv_base[1])
    relu5_3_Z = Flatten()(conv_base[2])
    UTX = Dense(512, activation="relu", name='UTX')(relu5_1_X)
    VTY = Dense(512, activation="relu", name='VTY')(relu5_2_Y)
    WTZ = Dense(512, activation="relu", name='WTZ')(relu5_3_Z)
    UTXoVTY = Multiply(name='UTXoVTY')([UTX, VTY])
    UTXoWTZ = Multiply(name='UTXoWTZ')([UTX, WTZ])
    VTYoWTZ = Multiply(name='VTYoWTZ')([VTY, WTZ])
    UTXoVTYoWTZ = Multiply(name='UTXoVTYoWTZ')([UTX, VTY, WTZ])
    concat = Concatenate()([UTXoVTY, UTXoWTZ, VTYoWTZ, UTXoVTYoWTZ])
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(concat))

    # Build the modelget_pred_indexes
    model = Model(name='MLPH_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


class Masked_Output(keras.layers.Layer):
    def __init__(self, M):
        super(Masked_Output, self).__init__()
        self.M = []
        for m in M:
            self.M.append(tf.convert_to_tensor(m, dtype=tf.float32))

    def build(self, input_shape):
        """Creates weights."""
        # Estimate the size of each output using the taxonomy.
        self.size_outputs = []
        self.size_outputs.append(len(self.M[0]))
        for m in self.M:
            self.size_outputs.append(len(m[0]))
        # Estimate the input size
        input_dims = []
        if isinstance(input_shape, list):
            for i in range(len(self.size_outputs)):
                input_dims.append(input_shape[i][1])
        else:
            for i in range(len(self.size_outputs)):
                input_dims.append(input_shape[1])
        # Create parameters W and B of the output.
        self.W = []
        self.b = []
        # self.W_mask = []
        # self.b_mask = []
        for i, (input_dim, size_output) in enumerate(zip(input_dims, self.size_outputs)):
            self.W.append(self.add_weight(name='W_' + str(i), shape=(input_dim, size_output),
                                          initializer="random_normal",
                                          trainable=True))
            self.b.append(self.add_weight(name='B_' + str(i), shape=(size_output,),
                                          initializer="zeros",
                                          trainable=True))
            # self.W_mask.append(self.add_weight(shape=(size_output, size_output),
            #                            initializer="random_normal",
            #                            trainable=True))
            # self.b_mask.append(self.add_weight(shape=(size_output,),
            #                            initializer="zeros",
            #                            trainable=True))

    def call(self, inputs):
        # Estimate the inputs.
        if not isinstance(inputs, list):
            inputs = [inputs for x in range(len(self.size_outputs))]

            # Compute the logits.
        z_list = []
        # z_mask_list = []

        for i in range(len(self.size_outputs)):
            z = tf.matmul(inputs[i], self.W[i]) + self.b[i]
            z_list.append(z)
            # z_mask_list.append(tf.matmul(tf.nn.relu(z), self.W_mask[i]) + self.b_mask[i])
        # Compute the masks.
        masks = []
        masks.append(tf.matmul(tf.nn.softmax(z_list[1]), tf.transpose(self.M[0])))
        for i in range(1, len(self.size_outputs) - 1):
            m_mid1 = tf.matmul(tf.nn.softmax(z_list[i - 1]), self.M[i - 1])
            # m_mid2 = tf.matmul(tf.nn.softmax(z_list[i + 1]), tf.transpose(self.M[i]))
            mask_sum = m_mid1
            masks.append(tf.math.minimum(mask_sum, 1))
        masks.append(tf.matmul(tf.nn.softmax(z_list[-2]), self.M[-1]))
        # Applying the masks and compute outputs.
        out = []
        out.append(tf.nn.softmax(z_list[0]))
        for i in range(1, len(self.size_outputs)):
            out.append(tf.nn.softmax(z_list[i] * masks[i]))
        return out

    def get_config(self):
        config = super(Masked_Output, self).get_config()
        # config.update({'M': self.M,
        #                'W': self.W,
        #                'b': self.b,
        #                #                   'W_mask': self.W_mask,
        #                #                   'b_mask': self.b_mask
        #                })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_Masked_Output_Net(num_classes: list,
                          image_size,
                          taxonomy,
                          conv_base='vgg19',
                          learning_rate=1e-5,
                          loss_weights=[],
                          mnets=False,
                          lam=0):
    # Conv base
    regularizer = tf.keras.regularizers.l2(lam)
    in_layer = Input(shape=image_size, name='main_input')
    if not mnets:
        conv_base = get_conv_base(conv_base, regularizer=regularizer)(in_layer)
        conv_base = Flatten()(conv_base)
        # outputs
        outputs = Masked_Output(taxonomy)(conv_base)
    else:
        conv_base_list = [get_conv_base(conv_base, regularizer=regularizer) for x in num_classes]
        for i in range(len(conv_base_list)):
            conv_base_list[i]._name = 'conv_base_mcnn_' + str(i)
            conv_base_list[i] = conv_base_list[i](in_layer)
            conv_base_list[i] = Flatten()(conv_base_list[i])
        # outputs
        outputs = Masked_Output(taxonomy)(conv_base_list)

    # Build the model
    model = Model(name='Masked_Output_Net',
                  inputs=in_layer,
                  outputs=outputs)
    loss = [keras.losses.SparseCategoricalCrossentropy() for x in num_classes]
    if len(loss_weights) == 0:
        loss_weights = [1 / len(num_classes) for x in num_classes]
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    return model


class nin_model(Model):

    def __init__(self):
        super(nin_model, self).__init__()

    def build(self, input_shape):
        self.Conv2D1 = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')
        self.Conv2D2 = Conv2D(filters=160, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D3 = Conv2D(filters=96, kernel_size=(1, 1), activation='relu', padding='same')

        self.Conv2D4 = Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same')
        self.Conv2D5 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D6 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')

        self.Conv2D7 = Conv2D(filters=192, kernel_size=(3, 3), activation='relu', padding='same')
        self.Conv2D8 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')
        self.Conv2D9 = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')

    def call(self, inputs):
        x = self.Conv2D1(inputs)
        x = self.Conv2D2(x)
        x = self.Conv2D3(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)
        x = Dropout(0.5)(x)
        x = self.Conv2D4(x)
        x = self.Conv2D5(x)
        x = self.Conv2D6(x)
        x = MaxPool2D(2, strides=2, padding='same')(x)
        x = Dropout(0.5)(x)
        x = self.Conv2D7(x)
        x = self.Conv2D8(x)
        x = self.Conv2D9(x)
        return GlobalAveragePooling2D()(x)


def get_conv_base(conv_base, regularizer=tf.keras.regularizers.l2(0)):
    if conv_base.lower() == 'vgg19':
        conv_base = VGG19(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'vgg16':
        conv_base = VGG16(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'nin':
        conv_base = nin_model()
    elif conv_base.lower() == 'resnet50':
        conv_base = ResNet50(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'xception':
        conv_base = Xception(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'efficientnetb0':
        conv_base = EfficientNetB0(weights='imagenet', include_top=False)
    else:
        return None
    for layer in conv_base.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return conv_base


if __name__ == '__main__':
    dataset = Cifar100()
    num_classes = [dataset.num_classes_l0, dataset.num_classes_l1, dataset.num_classes_l2]
    model = get_mnets(num_classes, dataset.image_size)
    model.summary()
