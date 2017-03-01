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

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import hp, STATUS_OK
import cPickle
import gzip


# load training and validation data
filename_train_set              = '../trainingData/train_set_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'
filename_train_validation_set   = '../trainingData/train_set_all_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'
filename_validation_set         = '../trainingData/validation_set_danAll_phonemeSeg_mfccBands_neighbor.pickle.gz'

with gzip.open(filename_train_set, 'rb') as f:
    X_train, Y_train = cPickle.load(f)

with gzip.open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation = cPickle.load(f)

with gzip.open(filename_validation_set, 'rb') as f:
    X_validation, Y_validation = cPickle.load(f)

# X_train = np.transpose(X_train)
Y_train             = to_categorical(Y_train)
Y_train_validation  = to_categorical(Y_train_validation)
Y_validation        = to_categorical(Y_validation)

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                      {'layers': 'three',}
                      ]),

            'units1': hp.uniform('units1', 64, 512),

            'dropout1': hp.uniform('dropout1', .25, .75),

            'batch_size' : hp.uniform('batch_size', 28, 128),

            'nb_epochs' :  500,
            'optimizer': hp.choice('optimizer',['adadelta','adam']),
            'activation': 'relu'
        }

def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X_train.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units1'], init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(output_dim=params['units1'], init="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout1']))

    model.add(Dense(29))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics = ['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    model.fit(X_train, Y_train,
              nb_epoch=params['nb_epochs'],
              batch_size=params['batch_size'],
              validation_data = (X_validation, Y_validation),
              callbacks=callbacks,
              verbose = 0)

    score, acc = model.evaluate(X_validation, Y_validation, batch_size = 128, verbose = 0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK}

def f_nn_model(node_size, dropout, i_d):
    model = Sequential()

    from keras.layers import Dense, Activation, Dropout

    model.add(Dense(output_dim=node_size, input_dim=i_d))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(node_size))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(output_dim=29))
    model.add(Activation("softmax"))

    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # best parameter model training

    model_0 = f_nn_model(511, 0.251, 400)

    print model_0.count_params()

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    hist = model_0.fit(X_train,
                    Y_train,
                    nb_epoch=500,
                    validation_data = (X_validation, Y_validation),
                    callbacks = callbacks,
                    batch_size=58)

    nb_epoch = len(hist.history['acc'])

    model_1 = f_nn_model(511, 0.251, 400)

    hist = model_1.fit(X_train_validation,
                    Y_train_validation,
                    nb_epoch=nb_epoch,
                    batch_size=58)

    model_1.save('./danAll/keras.dnn_2_optim_mfccBands_neighbor_all.h5')