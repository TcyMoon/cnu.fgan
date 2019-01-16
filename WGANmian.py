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
r"""Training / evaluation routine.

Example usage:

python main.py \
--model_name=cnn_2x_siamese \
--task_name=conditioned_classification \
--logdir=experiments/cnn_2x_siamese_experiment \
--source_dir=~/Downloads/holstep
"""
# !/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
from keras import layers
import tensorflow as tf

import conditioned_classification_models
import data_utils
import unconditioned_classification_models

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
# import matplotlib.pyplot as plt
import sys
# import mxnet
from keras.optimizers import RMSprop

import keras.backend as K
import numpy as np

# /home/user/TCY/holstep
# /Users/chenyangtang/python/holstep
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('source_dir',
                           '/home/user/TCY/holstep',
                           'Directory where the raw data is located.')
tf.app.flags.DEFINE_string('logdir',
                           '/home/user/TCY/deepmath-master/deepmath/holstep_baselines/log',
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

# def main(_):
#   logging.basicConfig(level=logging.DEBUG)
#   if not os.path.exists(FLAGS.logdir):
#     os.makedirs(FLAGS.logdir)
#
#   if FLAGS.tokenization == 'token':
#     use_tokens = True
#   elif FLAGS.tokenization == 'char':
#     use_tokens = False
#   else:
#     raise ValueError('Unknown tokenization mode:', FLAGS.tokenization)
#
#   # Parse the training and validation data.
#   parser = data_utils.DataParser(FLAGS.source_dir, use_tokens=use_tokens,
#                                  verbose=FLAGS.verbose)
#
#   # Print useful stats about the parsed data.
#   if FLAGS.verbose:
#     logging.info('Training data stats:')
#     parser.display_stats(parser.train_conjectures)
#     logging.info('---')
#     logging.info('Validation data stats:')
#     parser.display_stats(parser.val_conjectures)
#
#   voc_size = len(parser.vocabulary) + 1
#
#   # if FLAGS.task_name == 'conditioned_classification':
#   #   # Get the function for building the model, and the encoding to use.
#   #   make_model, encoding = conditioned_classification_models.MODELS.get(
#   #       FLAGS.model_name, None)
#   #   if not make_model:
#   #     raise ValueError('Unknown model:', FLAGS.model_name)
#   #
#   #   # Instantiate a generator that will yield batches of training data.
#   #   train_generator = parser.training_steps_and_conjectures_generator(
#   #       encoding=encoding, max_len=FLAGS.max_len,
#   #       batch_size=FLAGS.batch_size)
#   #
#   #   # Instantiate a generator that will yield batches of validation data.
#   #   val_generator = parser.validation_steps_and_conjectures_generator(
#   #       encoding=encoding, max_len=FLAGS.max_len,
#   #       batch_size=FLAGS.batch_size)
#   #
#   # elif FLAGS.task_name == 'unconditioned_classification':
#   make_model, encoding = unconditioned_classification_models.MODELS.get(
#         FLAGS.model_name, None)
#   # if not make_model:
#   #     raise ValueError('Unknown model:', FLAGS.model_name)
#   train_generator = parser.training_steps_generator(
#         encoding=encoding, max_len=FLAGS.max_len,
#         batch_size=FLAGS.batch_size)
#
#   val_generator = parser.validation_steps_generator(
#         encoding=encoding, max_len=FLAGS.max_len,
#         batch_size=FLAGS.batch_size)
#
#   # else:
#   #   raise ValueError('Unknown task_name:', FLAGS.task_name)
#
#   if FLAGS.checkpoint_path:
#     # Optionally load an existing saved model.
#     model = keras.models.load_model(FLAGS.checkpoint_path)
#   else:
#     # Instantiate a fresh model.
#     model = make_model(voc_size, FLAGS.max_len)
#     model.summary()
#     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
#                   loss='binary_crossentropy',
#                   metrics=['acc'])
#
#   # Define a callback for saving the model to the log directory.
#   checkpoint_path = os.path.join(FLAGS.logdir, FLAGS.model_name + '.h5')
#   checkpointer = keras.callbacks.ModelCheckpoint(
#       checkpoint_path, save_best_only=True)
#   earlystopping = keras.callbacks.EarlyStopping(monitor='acc', patience=0, verbose=0, mode='auto')
#
#   # Define a callback for writing TensorBoard logs to the log directory.
#   tensorboard_vis = keras.callbacks.TensorBoard(log_dir=FLAGS.logdir)
#
#   logging.info('Fit model...')
#   history = model.fit_generator(train_generator,
#                                 steps_per_epoch=FLAGS.samples_per_epoch,
#                                 validation_data=val_generator,
#                                 epochs=40,
#                                 validation_steps=FLAGS.val_samples,
#                                 use_multiprocessing=True,
#                                 workers=FLAGS.data_parsing_workers,
#                                 verbose=FLAGS.verbose,
#                                 callbacks=[checkpointer, tensorboard_vis, earlystopping])
#
#   # Save training history to a JSON file.
#   f = open(os.path.join(FLAGS.logdir, 'history.json'), 'w')
#   f.write(json.dumps(history.history))
#   f.close()


#
class PreGAN():
    def __init__(self):
        # Input shape
        # self.img_shape = (512,)
        self.latent_dim = 87
        self.input_dim = 512
        self.n_critic = 5
        self.voc_size = 0
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)
        self.parser = data_utils.DataParser(FLAGS.source_dir,
                                            use_tokens=False,
                                            verbose=FLAGS.verbose)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        print("+++++++++++++++")
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        #tensorboard_vis = keras.callbacks.TensorBoard(log_dir=FLAGS.logdir)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        #
        #
        model.add(Embedding(input_dim=87, output_dim=256, input_length=self.input_dim))
        # model.add(Dense(256*3, activation="relu"))
        # model.add(Reshape((3, 256)))
        # model.add(UpSampling2D())
        model.add(Conv1D(256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        model.add(MaxPooling1D(3))
        # model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(MaxPooling1D(5))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation("relu"))
        # model.add(MaxPooling1D(5))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Activation("tanh"))
        # model.add(LSTM(32))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(256, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(512, kernel_size=3, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))

        model.summary()
        # noise = layers.Input(shape=(max_len,), dtype='int32')
        # x = layers.Embedding(
        #     output_dim=256,
        #     input_dim=voc_size,
        #     input_length=max_len)(noise)
        # x = layers.Convolution1D(256, 7, activation='relu')(x)
        # x = layers.MaxPooling1D(3)(x)
        # x = layers.Convolution1D(256, 7, activation='relu')(x)
        # x = layers.MaxPooling1D(5)(x)
        # embedded_statement = layers.LSTM(256)(x)
        #
        # x = layers.Dense(256, activation='relu')(embedded_statement)
        # img = layers.Dropout(dropout)(x)
        # prediction = layers.Dense(1, activation='sigmoid')(x)

        noise = Input(shape=(self.latent_dim,))

        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # model.add(Embedding(input_dim=87,output_dim=self.input_dim,input_length=input_dim))
        model.add(Dense(self.input_dim * 87, activation="relu", input_shape=(self.input_dim,)))

        model.add(Reshape((87*16,32)))
        model.add(Conv1D(32, kernel_size=5,input_shape = (32,),activation='relu'))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        # model.add(MaxPooling1D(3))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, padding="same"))
        # model.add(ZeroPadding1D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv1D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv1D(512,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #model.add(LSTM(256))
        # model.add(Flatten())

        model.add(Dense(1))
        # model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=(self.input_dim,))
        # print(img.get_shape().as_list())
        validity = model(img)
        model.save('myDiscriminator_model.h5')
        return Model(img, validity)

    def train(self, epochs, batch_size, save_interval=50):

        # (valencode, _), (_, _) = parser.draw_batch_of_steps_in_order(self, conjecture_index=0,
        #                                                     step_index=0,
        #                                                     split='train',
        #                                                     encoding='integer',
        #                                                     max_len=256, batch_size=128)
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1

        # Adversarial ground truths

        # valid = np.ones((batch_size, 1))
        # fake = np.zeros((batch_size, 1))
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            X_train, _ = self.parser.draw_random_batch_of_steps(
                'train', 'integer', 512, 128)

            self.voc_size = len(self.parser.vocabulary) + 1
            X_train = (X_train.astype(np.float32) - 86.5) / 86.5
            print(X_train.shape)
            # X_train = np.expand_dims(X_train, axis=2)
            #  Train Discriminator
            # ---------------------
            for _ in range(self.n_critic):

                # Select a random half of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise and generate a batch of new images
                # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                noise = tf.truncated_normal(
                    shape=[batch_size, self.latent_dim], mean=43, stddev=21.5, dtype=tf.float32)

                with tf.Session():
                    noise_np = noise.eval()

                gen_imgs = self.generator.predict(noise_np)
                # Train the discriminator (real classified as ones and generated as
                # zeros)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as
            # real)

            g_loss = self.combined.train_on_batch(noise_np, valid)
            # g_loss = self.combined.evaluate(noise_np,valid,32)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, 1 - d_loss[0], 100 * d_loss[1], 1 - g_loss))

            # If at save interval => save generated image samples
            # if epoch % save_interval == 0:
            # self.save_imgs(epoch)

            # def save_imgs(self, epoch):
            #     r, c = 5, 5
            #     noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            #     gen_imgs = self.generator.predict(noise)
            #
            #     # Rescale images 0 - 1
            #     gen_imgs = 0.5 * gen_imgs + 0.5
            #
            #     fig, axs = plt.subplots(r, c)
            #     cnt = 0
            #     for i in range(r):
            #         for j in range(c):
            #             axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            #             axs[i,j].axis('off')
            #             cnt += 1
            #     fig.savefig("images/mnist_%d.png" % epoch)
            #     plt.close()


if __name__ == '__main__':
    pregan = PreGAN()
    pregan.train(epochs=4000, batch_size=32, save_interval=50)
