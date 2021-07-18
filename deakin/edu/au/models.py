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
    GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import numpy as np
from deakin.edu.au.data import Cifar100


def get_MLPH_model(num_classes: list, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
                   learning_rate=1e-5, pi=0.5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))

    # Build the model
    model = Model(name='MLPH_model',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=[loss, loss],
                  loss_weights=[1 - pi, pi],
                  metrics=['accuracy'])
    return model


def get_BCNN1(num_classes: list, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
              learning_rate=1e-5, pi=0.5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layers = []
    for idx, v in enumerate(num_classes):
        if len(out_layers) == 0:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))
        else:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(out_layers[-1]))
    # Build the model
    model = Model(name='Model_BCNN1',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=[loss, loss],
                  loss_weights=[1 - pi, pi],
                  metrics=['accuracy'])
    return model


def get_BCNN2(num_classes: list, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
              learning_rate=1e-5, pi=0.5):
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)

    # create output layers
    out_layers = []
    for idx, v in enumerate(reversed(num_classes)):
        idx = len(num_classes) - idx - 1
        if len(out_layers) == 0:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(conv_base))
        else:
            out_layers.append(Dense(v, activation="softmax", name='out_level_' + str(idx))(out_layers[-1]))

    # Build the model
    model = Model(name='Model_BCNN2',
                  inputs=in_layer,
                  outputs=out_layers)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=[loss, loss],
                  loss_weights=[1 - pi, pi],
                  metrics=['accuracy'])
    return model


def get_BCNN3(num_classes: list, image_size, conv_base=VGG19(include_top=False, weights="imagenet"),
              learning_rate=1e-5, pi=0.5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = conv_base(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    logits_layers = []
    out_layers = []
    for idx, v in enumerate(num_classes):
        if len(logits_layers) == 0:
            logits = Dense(v, name='logits_c')(conv_base)

        else:
            logits = Dense(v, name='logits_c')(logits_layers[-1])
        logits_layers.append(logits)
        out_layers.append(Activation(v, activation="softmax", name='out_level_' + str(idx))(logits))


    # coarse output
    logits_c = Dense(num_classes_c, name='logits_c')(conv_base)
    out_c = Activation(keras.activations.softmax, name='out_c')(logits_c)

    # fine output
    relu_c = Activation(keras.activations.relu, name='relu_c')(logits_c)
    out_f = Dense(num_classes_f, activation="softmax", name='out_f')(relu_c)

    # Build the model
    model = Model(name='Model_4',
                  inputs=in_layer,
                  outputs=[out_c, out_f])
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=[loss, loss],
                  loss_weights=[1 - pi, pi],
                  metrics=['accuracy'])
    return model


# model = get_model4()
# model.summary()
# plot_model(model)


if __name__ == '__main__':
    dataset = Cifar100()
    num_classes = [dataset.num_classes_c, dataset.num_classes_f]
    model = get_BCNN1(num_classes, dataset.image_size)
    model.summary()
    # plot_model(model)
