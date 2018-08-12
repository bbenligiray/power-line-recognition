# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from .imagenet_utils import decode_predictions
from .imagenet_utils import preprocess_input
from .imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), weight_decay=0):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1',
                      kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(weights='imagenet', input_tensor=None, weight_decay=0,
            no_cats=2, activation='softmax'):
    """
    Builds the entire model, excluding the final fully connected layer.
    Adds a randomly initialized, fully connected layer to the end.
    Feed the input tensor as thus:
        input_tensor=keras.layers.Input(shape=(224, 224, 3))
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), weight_decay=weight_decay)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    model = Model(img_input, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')

    x = Flatten()(model.output)
    x = Dense(no_cats, activation = activation,
        kernel_regularizer=l2(weight_decay), name = 'fc_final')(x)
    model = Model(inputs = model.input, outputs = x)

    return model


def ResNet50_tail(input_tensor=None, weight_decay=0,
                stage='final', no_cats=2, activation='softmax'):
    """
    Similar to ResNet50(). Builds only the tail of the model, starting from 'stage'.
    Cannot use ImageNet weights. However, you can copy them after building the model.
    """

    img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if not stage in ['final', '5']:
        raise ValueError('Invalid stage')

    if stage == '5':
        x = conv_block(img_input, 3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay)

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = Flatten()(x)
        x = Dense(no_cats, activation = activation,
            kernel_regularizer=l2(weight_decay), name = 'fc_final')(x)
    elif stage == 'final':
        x = Flatten()(img_input)
        x = Dense(no_cats, activation = activation,
            kernel_regularizer=l2(weight_decay), name = 'fc_final')(x)
    else:
        raise ValueError('Invalid stage')

    model = Model(img_input, x)

    return model


def get_no_layers(stage):
    """
    returns the total number of layers the model has up to a stage
    for example, if stage == 5, returns len(stage 1, 2, 3, 4)
    """
    if stage == 'final':
        return 175
    elif stage == '5':
        return 142
    else:
        raise ValueError('Invalid stage')
