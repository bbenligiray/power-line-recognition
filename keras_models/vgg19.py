# -*- coding: utf-8 -*-
"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from .imagenet_utils import decode_predictions
from .imagenet_utils import preprocess_input
from .imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(weights='imagenet', input_tensor=None, weight_decay=0,
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
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4',
               kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
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


def VGG19_tail(input_tensor=None, weight_decay=0,
                stage='final', no_cats=2, activation='softmax'):
    """
    Similar to VGG19(). Builds only the tail of the model, starting from 'stage'.
    Cannot use ImageNet weights. However, you can copy them after building the model.
    """

    img_input = input_tensor

    if not stage in ['final', '5']:
        raise ValueError('Invalid stage')

    if stage == '5':
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
                  kernel_regularizer=l2(weight_decay))(img_input)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
                  kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
                  kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4',
                  kernel_regularizer=l2(weight_decay))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

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
        return 22
    elif stage == '5':
        return 17
    else:
        raise ValueError('Invalid stage')
