# -*- coding: utf-8 -*-
'''
 * Copyright (C) 2017  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of EUSIPCO2017 phoneme classification
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
'''


from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

import cPickle,gzip

# load training and validation data
filename_train_validation_set   = '../trainingData/train_set_all_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
filename_train_set              = '../trainingData/train_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'
filename_validation_set         = '../trainingData/validation_set_laosheng_phonemeSeg_mfccBands2D.pickle.gz'

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_train_set, 'rb') as f:
    X_train, Y_train = cPickle.load(f)

with gzip.open(filename_validation_set, 'rb') as f:
    X_validation, Y_validation = cPickle.load(f)

# X_train = np.transpose(X_train)
Y_train_validation  = to_categorical(Y_train_validation)
Y_train             = to_categorical(Y_train)
Y_validation        = to_categorical(Y_validation)

def squareFilterChoiCNN():

    nlen            = 21
    filter_density  = 1
    channel_axis    = 1
    reshape_dim     = (1, 80, nlen)
    input_dim       = (80, nlen)

    model_1 = Sequential()
    model_1.add(Reshape(reshape_dim, input_shape=input_dim))
    model_1.add(Convolution2D(60 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='th'))

    model_1.add(Convolution2D(84 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(2, 1), border_mode='valid', dim_ordering='th'))


    model_1.add(Convolution2D(84 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(2, 1), border_mode='valid', dim_ordering='th'))

    model_1.add(Convolution2D(84 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(ELU())
    model_1.add(MaxPooling2D(pool_size=(2, 1), border_mode='valid', dim_ordering='th'))

    model_1.add(Convolution2D(60 * filter_density, 3, 3, border_mode='valid', input_shape=reshape_dim, dim_ordering='th'))
    model_1.add(BatchNormalization(axis=channel_axis, mode=0))
    model_1.add(ELU())
    model_1.add(Flatten())

    model_1.add(Dense(output_dim=29))
    model_1.add(Activation("softmax"))

    optimizer = Adam()

    model_1.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])

    return model_1

def train_model(file_path_model):
    """
    train final model save to model path
    """
    model_0 = squareFilterChoiCNN()

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]

    hist = model_0.fit(X_train,
              Y_train,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              nb_epoch=500,
              batch_size=128)

    nb_epoch = len(hist.history['acc'])

    model_1 = squareFilterChoiCNN()

    model_1.fit(X_train_validation,
                    Y_train_validation,
                    nb_epoch=nb_epoch,
                    batch_size=128)

    model_1.save(file_path_model)

file_path_model = './qmLonUpf/laosheng/keras.cnn_choi_mfccBands_2D_all_optim.h5'
train_model(file_path_model)

