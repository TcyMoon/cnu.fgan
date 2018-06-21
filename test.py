from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import keras
import tensorflow as tf

import conditioned_classification_models
import data_utils
import unconditioned_classification_models

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import os
#
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_dir',
                           '/Users/chenyangtang/python/holstep',
                           'Directory where the raw data is located.')
tf.app.flags.DEFINE_string('logdir',
                           '/tmp/hol',
                           'Base directory for saving models and metrics.')
tf.app.flags.DEFINE_string('model_name',
                           'cnn_2x',
                           'Name of model to train.')
tf.app.flags.DEFINE_string('task_name',
                           'unconditioned_classification',
                           'Name of task to run: "conditioned_classification" '
                           'or "unconditioned_classification".')
tf.app.flags.DEFINE_string('tokenization',
                           'char',
                           'Type of statement tokenization to use: "char" or '
                           '"token".')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'Size of a batch.')
tf.app.flags.DEFINE_integer('max_len', 512,
                            'Maximum length of input statements.')
tf.app.flags.DEFINE_integer('samples_per_epoch', 12800,
                            'Number of random step statements to draw for '
                            'training at each epoch.')
tf.app.flags.DEFINE_integer('val_samples', 246912,
                            'Number of (ordered) step statements to draw for '
                            'validation.')
tf.app.flags.DEFINE_integer('epochs', 40,
                            'Number of epochs to train.')
tf.app.flags.DEFINE_integer('verbose', 1,
                            'Verbosity mode (0, 1 or 2).')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Path to checkpoint to (re)start from.')
tf.app.flags.DEFINE_integer('data_parsing_workers', 4,
                            'Number of threads to use to generate input data.')


def main():
    print("start")
    parser = data_utils.DataParser(FLAGS.source_dir,
                                           use_tokens=False,
                                           verbose=FLAGS.verbose)
    print (2)
    X_train, _ = parser.draw_random_batch_of_steps(
              'train', 'integer', FLAGS.max_len, FLAGS.batch_size)
    print (2)
    print(X_train)

if __name__ == '__main__':
    main()
