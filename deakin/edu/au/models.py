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
    GlobalAveragePooling2D, Multiply, Concatenate
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.applications import VGG19,VGG16
from deakin.edu.au.data import Cifar100
import deakin.edu.au.metrics as metrics
import numpy as np
import tensorflow as tf


class performance_callback(keras.callbacks.Callback):
    def __init__(self, X, y, taxo):
        self.X = X
        self.y = y
        self.taxo = taxo

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        # y_pred = get_pred_indexes(y_pred)
        accuracy = metrics.get_top_k_taxonomical_accuracy(self.y, y_pred)
        exact_match = metrics.get_exact_match(self.y, y_pred)
        consistency = metrics.get_consistency(y_pred, self.taxo)
        print('-' * 100)
        print(f"epoch={epoch + 1}, ", end='')
        print(f"Exact Match = {exact_match:.4f}, ", end='')
        for i in range(len(accuracy)):
            print(f"accuracy level_{i} = {accuracy[i]:.4f}, ", end='')
        print(f"Consistency = {consistency:.4f}")
        print('-' * 100)
        print('')


def get_mout_model(num_classes: list,
                   image_size,
                   conv_base='vgg19',
                   learning_rate=1e-5,
                   loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base)(in_layer)
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
              loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base)(in_layer)
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
              loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base)(in_layer)
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
              loss_weights=[]):
    in_layer = Input(shape=image_size, name='main_input')
    # Conv base
    conv_base_list = []
    if conv_base.lower() == 'vgg19':
        conv_base_list = [VGG19(include_top=False, weights="imagenet") for x in num_classes]
    elif conv_base.lower() == 'nin':
        conv_base_list = [nin_model() for x in num_classes]
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
                       learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base)(in_layer)
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
                         learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size)
    conv_base = get_conv_base(conv_base)(in_layer)
    conv_base = Flatten()(conv_base)
    # create output layers
    out_layer = Dense(num_classes, activation="softmax")(conv_base)
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
                   loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = VGG19(include_top=False, weights="imagenet")
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
        input_dim = input_shape[1]

        # Estimate the size of each output using the taxonomy.
        self.size_outputs = []
        self.size_outputs.append(len(self.M))
        for m in self.M:
            self.size_outputs.append(len(m[0]))
            # Create parameters W and B of the output.
        self.W = []
        self.b = []
        #         self.W_mask = []
        #         self.b_mask = []
        for size_output in self.size_outputs:
            self.W.append(self.add_weight(shape=(input_dim, size_output),
                                          initializer="random_normal",
                                          trainable=True))
            self.b.append(self.add_weight(shape=(size_output,),
                                          initializer="zeros",
                                          trainable=True))

    #             self.W_mask.append(self.add_weight(shape=(size_output, size_output),
    #                                        initializer="random_normal",
    #                                        trainable=True))
    #             self.b_mask.append(self.add_weight(shape=(size_output,),
    #                                        initializer="zeros",
    #                                        trainable=True))

    def call(self, inputs):
        # Compute the logits.
        z_list = []
        #         z_mask_list = []

        for i in range(len(self.size_outputs)):
            z = tf.matmul(inputs, self.W[i]) + self.b[i]
            z_list.append(z)
        #             z_mask_list.append(tf.matmul(tf.nn.relu(z), self.W_mask[i]) + self.b_mask[i])
        # Compute the masks.
        masks = []
        masks.append(tf.matmul(tf.nn.softmax(z_list[1]), tf.transpose(self.M[0])))
        for i in range(1, len(self.size_outputs) - 1):
            m_mid1 = tf.matmul(tf.nn.softmax(z_list[i - 1]), self.M[i - 1])
            m_mid2 = tf.matmul(tf.nn.softmax(z_list[i + 1]), tf.transpose(self.M[i]))
            mask_sum = m_mid1 + m_mid2
            masks.append(tf.math.minimum(mask_sum, 1))
        masks.append(tf.matmul(tf.nn.softmax(z_list[-2]), self.M[-1]))
        # Applying the masks and compute outputs.
        out = []
        for i in range(len(self.size_outputs)):
            out.append(tf.nn.softmax(z_list[i] * masks[i]))
        return out

    def get_config(self):
        config = {'M': self.M,
                  'W': self.W,
                  'b': self.b,
                  #                   'W_mask': self.W_mask,
                  #                   'b_mask': self.b_mask
                  }
        base_config = super(Masked_Output, self).get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_Masked_Output_Net(num_classes: list,
                          image_size,
                          taxonomy,
                          conv_base='vgg19',
                          learning_rate=1e-5,
                          loss_weights=[]):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    conv_base = get_conv_base(conv_base)(in_layer)
    conv_base = Flatten()(conv_base)
    # outputs
    outputs = Masked_Output(taxonomy)(conv_base)
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


def get_conv_base(conv_base):
    if conv_base.lower() == 'vgg19':
        conv_base = VGG19(include_top=False, weights="imagenet")
    elif conv_base.lower() == 'nin':
        conv_base = nin_model()
    return conv_base


def get_hdcnn_coarse_classifier(num_classes: list, image_size, level, learning_rate=1e-5):
    # Conv base
    in_layer = Input(shape=image_size, name='main_input')
    
    model = VGG16(include_top=False, weights="imagenet",input_tensor=in_layer)
    
    #Freeze parameters
    for i in range(len(model.layers)):
        
        model.layers[i].trainable=False
        
    x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(model.layers[-6].output)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_coarse = Dense(num_classes[level], activation='softmax',name='predictions')(x)
    
    # Build the model
    model_c = Model(name='hdcnn_model',inputs=in_layer,outputs=out_coarse)
    
    for i in range(len(model_c.layers)-5):
        model_c.layers[i].set_weights(model.layers[i].get_weights())
        
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model_c.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    
    return model_c

def get_hdcnn_fine_classifiers(num_classes: list, image_size, level, learning_rate=1e-5):
    
    in_layer = Input(shape=image_size, name='main_input')
    
    model = VGG16(include_top=False, weights="imagenet",input_tensor=in_layer)
    
    #Freeze parameters
    for i in range(len(model.layers)):
        
        model.layers[i].trainable=False
        
    x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(model.layers[-6].output)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D( 512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out_fine = Dense(num_classes[level], activation='softmax',name='predictions')(x)


    model_fine = Model(inputs=in_layer,outputs=out_fine)
    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model_fine.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    
    for i in range(len(model_fine.layers)-5):
        model_fine.layers[i].set_weights(model.layers[i].get_weights())
        
    return model_fine

def HD_CNN(dataset,num_classes,batch_size,epochs_c,epochs_f):

    #coarse_lv: (int) number of coarse classes. Should be less than fine_lv
    #fine_lv: (int) number of fine classes. Should be more than coarse_lv
    # epochs_c: number of epochs for coarse model
    # epochs_f: number of epochs for fine models
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    predictions=[]
    
    #we run the algorithm HD-CNN for every pair of labels existing in num_classes
    for l in range(len(num_classes)-1):
    
    
        coarse_lv=l
        fine_lv=l+1
        
        coarse_categories = num_classes[coarse_lv]
        fine_categories = num_classes[fine_lv]
        
        y_c_test=dataset.y_test[coarse_lv]
        y_f_test=dataset.y_test[fine_lv]
        
        fine2coarse = np.zeros((fine_categories,coarse_categories))
        for i in range(coarse_categories):
            index = np.where(y_c_test[:,0] == i)[0]
            fine_cat = np.unique([y_f_test[j,0]  for j in index])
            for j in fine_cat:
                fine2coarse[j,i] = 1
        
        #get coarse classifier
        coarse_model=get_hdcnn_coarse_classifier(num_classes, dataset.image_size,level=coarse_lv)
        
        coarse_model.summary()
        
        #Training coarse model
        coarse_model.fit(dataset.X_train, 
                         dataset.y_train[coarse_lv],
                         validation_data = (dataset.X_val, dataset.y_val[coarse_lv]),
                         batch_size=batch_size, 
                         epochs=epochs_c,
                         callbacks=[early_stopping_callback])
        
           
        
        for i in range(coarse_categories):
            
            # Get all training data for the coarse category
            
            #y_train=dataset.y_train[fine_lv]
            y_train_hot=metrics.one_hot(dataset.y_train[fine_lv])
        
            #get train labels that are inside each coarse classifier
            ix = np.where([(y_train_hot[:,:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
            x_tix = dataset.X_train[ix]
            y_train = dataset.y_train[fine_lv]
            y_tix = y_train[ix]
            
            #Definea model for each fine classifier
            fine_model = get_hdcnn_fine_classifiers(num_classes, dataset.image_size,level=fine_lv)
           
        
            fine_model.fit(x_tix, y_tix, 
                           batch_size=batch_size, 
                           epochs=epochs_f
                           )
            
            fine_model.save_weights('fine_models/fine_model'+str(i))
            
           
        #Evaluation mode
            
        y_test_hot=metrics.one_hot(dataset.y_test[fine_lv])
            
        yh = np.zeros([y_test_hot.shape[0],y_test_hot.shape[2]])
        
        #evaluate coarse model
        yh_c = coarse_model.predict(dataset.X_test)
        
        for i in range(coarse_categories):
            
            #Coarse predictions
            
            print("Evaluating Fine Classifier: ", str(i))
            
            fine_model.load_weights('fine_models/fine_model'+str(i))
            
            #evaluate fine models
            fine_predictions=fine_model.predict(dataset.X_test)
            
            #calculate final predictions
            yh += np.multiply(yh_c[:,i].reshape((len(dataset.y_test[fine_lv])),1),fine_predictions)
            
        yh_c_index=np.argmax(yh_c,axis=1)
        yh_index=np.argmax(yh,axis=1)
    
        #We store the first coarse classification 
        if len(predictions) == 0:
            predictions.append(yh_c_index)
        
        #After that, we only store the average prediction. The result is equal 
        #to the number of num_classes
        predictions.append(yh_index)
        
    return predictions


if __name__ == '__main__':
    dataset = Cifar100()
    num_classes = [dataset.num_classes_l0, dataset.num_classes_l1, dataset.num_classes_l2]
    model = get_mnets(num_classes, dataset.image_size)
    model.summary()
