# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Models for classifying proof steps with conditioning on the conjecture."""
# -*- coding: UTF-8 -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function

from keras import layers
from keras.models import Model
from keras import regularizers

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding1D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import AveragePooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
bn_axis = -1
def identity_block(input_tensor, kernel_size, filters, stage, block):
    
    filters1= filters
    filters2= filters
    filters3 = filters
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block):
    
    filters1= filters
    filters2= filters
    filters3 = filters
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
               
    x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
               
    shortcut = Conv1D(filters3, 1, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def cnn_2x_siamese(voc_size, max_len, dropout=0.5):

    # Determine proper input shape
    #input_shape = _obtain_input_shape(input_shape,
    #                                 default_size=224,
    #                                  min_size=197,
    #                                 data_format=K.image_data_format(),
    #                                 include_top=include_top)


    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')
    
    x = layers.Embedding(
                         output_dim=256,
                         input_dim=voc_size,
                         input_length=max_len)(pivot_input)

#if input_tensor is None:
#    img_input = Input(shape=input_shape)
        #    else:
        #        if not K.is_keras_tensor(input_tensor):
        #            img_input = Input(tensor=input_tensor, shape=input_shape)
                #       else:
#            img_input = input_tensor
   

    x = ZeroPadding1D(padding=1)(x)
    x = Conv1D(64, 7, strides=2, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)
    
    x = conv_block(x, 3, 32, stage=2, block='a')
    x = identity_block(x, 3, 32, stage=2, block='b')
    x = identity_block(x, 3, 32, stage=2, block='c')
    
    x = conv_block(x, 3, 64, stage=3, block='a')
    x = identity_block(x, 3, 64, stage=3, block='b')
    x = identity_block(x, 1, 64, stage=3, block='c')
    x = identity_block(x, 3, 64, stage=3, block='d')
    
    x = conv_block(x, 3, 128, stage=4, block='a')
    x = identity_block(x, 3, 128, stage=4, block='b')
    x = identity_block(x, 3, 128, stage=4, block='c')
    x = identity_block(x, 3, 128, stage=4, block='d')
    x = identity_block(x, 3, 128, stage=4, block='e')
    x = identity_block(x, 3, 128, stage=4, block='f')
    
    x = conv_block(x, 3, 256, stage=5, block='a')
    x = identity_block(x, 3, 256, stage=5, block='b')
    x = identity_block(x, 3, 256, stage=5, block='c')
    
    x = AveragePooling1D(7, name='avg_pool')(x)

    embedded_pivot = layers.LSTM(256)(x)
    

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)

    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model([pivot_input, statement_input], prediction)
    
    return model


def cnn_2x_lstm_siamese(voc_size, max_len, dropout=0.2):
    """Two siamese branches, each embedding a statement.

    Binary classifier on top.

    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
      dropout: Fraction of units to drop.
    Returns:
      A Keras model instance.
    """
    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(pivot_input)
        
    x = layers.Conv1D(32, 7, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(32, 7, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2,strides=2)(x)

    x = layers.Conv1D(64, 5, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(64, 5, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2,strides=2)(x)

    x = layers.Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2,strides=2)(x)

    x = layers.Conv1D(256, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.ZeroPadding1D(padding=1)(x)
    x = layers.Conv1D(256, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2,strides=2)(x)

    embedded_pivot = layers.LSTM(256)(x)

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)
    
    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='relu')(x)

    model = Model([pivot_input, statement_input], prediction)
    return model


def embedding_logreg_siamese(voc_size, max_len, dropout=0.5):
    """Two siamese branches, each embedding a statement.

    Binary classifier on top.

    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
      dropout: Fraction of units to drop.
    Returns:
      A Keras model instance.
    """
    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(pivot_input)
    x = layers.Activation('relu')(x)
    embedded_pivot = layers.Flatten()(x)

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)

    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dropout(dropout)(concat)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model([pivot_input, statement_input], prediction)
    return model


# Contains both the model definition function and the type of encoding needed.
MODELS = {
    'cnn_2x_siamese': (cnn_2x_siamese, 'integer'),
    'cnn_2x_lstm_siamese': (cnn_2x_lstm_siamese, 'integer'),
    'embedding_logreg_siamese': (embedding_logreg_siamese, 'integer'),
}
