#!/usr/bin/env python
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

import os
import numpy as np
import pickle
import essentia.standard as ess
from sklearn import preprocessing

from parameters import *
from phonemeMap import *
from textgridParser import syllableTextgridExtraction
from trainTestSeparation import getRecordingNames
from phonemeSampleCollection import getFeature,getMFCCBands1D,getMFCCBands2D,featureReshape
from phonemeClassification import PhonemeClassification
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


def doClassification():
    """
    1. collect features from test set
    2. predict by GMM or DNN models
    3. save the prediction
    :return: prediction of GMM and DNN model
    """

    phone_class = PhonemeClassification()
    phone_class.create_gmm(gmmModel_path)

    mfcc_all = np.array([])
    mfccBands1D_all = np.array([])
    mfccBands2D_all = np.array([])

    y_true = []

    for recording in getRecordingNames('TEST', dataset):
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        # plotAudio(audio,15,16)

        print 'calculating mfcc and mfcc bands ... ', recording
        mfcc                = getFeature(audio, d=True, nbf=False)
        mfccBands1D         = getMFCCBands1D(audio, nbf=True)
        mfccBands2D         = getMFCCBands2D(audio, nbf=True)
        mfccBands2D         = np.log(10000*mfccBands2D+1)

        # scale mfccBands1D for dnn acoustic models
        mfccBands1D_std         = preprocessing.StandardScaler().fit_transform(mfccBands1D)

        # scale mfccBands2D for cnn acoustic models
        scaler                  = pickle.load(open(scaler_path, 'rb'))
        mfccBands2D_std           = scaler.transform(mfccBands2D)


        for ii,pho in enumerate(nestedPhonemeLists):

            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))

            # MFCC feature
            sf = round(pho[0][0]*fs/hopsize)
            ef = round(pho[0][1]*fs/hopsize)

            # mfcc syllable
            mfcc_s      = mfcc[sf:ef,:]
            mfccBands_s     = mfccBands2D[sf:ef,:]
            mfccBands1D_s_std = mfccBands1D_std[sf:ef,:]
            mfccBands2D_s_std = mfccBands2D_std[sf:ef,:]


            if len(mfcc_all):
                mfcc_all        = np.vstack((mfcc_all,mfcc_s))
                mfccBands1D_all   = np.vstack((mfccBands1D_all,mfccBands1D_s_std))
                mfccBands2D_all   = np.vstack((mfccBands2D_all,mfccBands2D_s_std))
            else:
                mfcc_all        = mfcc_s
                mfccBands1D_all   = mfccBands1D_s_std
                mfccBands2D_all   = mfccBands2D_s_std

            # print mfcc_all.shape, mfccBands2D_all.shape

            ##-- parsing y_true
            y_true_s = []
            for ii_p, p in enumerate(pho[1]):
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]
                index_key = dic_pho_label[key]
                y_true_s += [index_key]*int(round((p[1]-p[0])/hopsize_t))

            print len(y_true_s), mfcc_s.shape[0]

            if len(y_true_s) > mfcc_s.shape[0]:
                y_true_s = y_true_s[:mfcc_s.shape[0]]
            elif len(y_true_s) < mfcc_s.shape[0]:
                y_true_s += [y_true_s[-1]]*(mfcc_s.shape[0]-len(y_true_s))

            y_true += y_true_s

    phone_class.mapb_gmm(mfcc_all)
    obs_gmm     = phone_class.mapb_gmm_getter()
    y_pred_gmm  = phone_class.prediction(obs_gmm)


    mfccBands2D_all   = featureReshape(mfccBands2D_all)

    phone_class.mapb_keras(mfccBands2D_all, kerasModels_jordi_path, jordi=True)
    obs_cnn_jordi   = phone_class.mapb_keras_getter()
    y_pred_jordi    = phone_class.prediction(obs_cnn_jordi)

    phone_class.mapb_keras(mfccBands2D_all, kerasModels_choi_path)
    obs_cnn_choi = phone_class.mapb_keras_getter()
    y_pred_choi  = phone_class.prediction(obs_cnn_choi)

    phone_class.mapb_keras(mfccBands1D_all, kerasModels_dnn_path)
    obs_dnn     = phone_class.mapb_keras_getter()
    y_pred_dnn  = phone_class.prediction(obs_dnn)

    np.save('./trainingData/y_pred_gmm.npy',y_pred_gmm)
    np.save('./trainingData/y_pred_jordi.npy',y_pred_jordi)
    np.save('./trainingData/y_pred_choi.npy',y_pred_choi)
    np.save('./trainingData/y_pred_dnn.npy',y_pred_dnn)

    np.save('./trainingData/y_true.npy',y_true)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # print(cm)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion_matrix(pred_path, true_path, title):
    y_pred = np.load(pred_path)
    y_true = np.load(true_path)

    label = []
    for ii in xrange(len(dic_pho_label_inv)):
        label.append(dic_pho_label_inv[ii])

    cm = confusion_matrix(y_true,y_pred)

    plt.figure()
    plot_confusion_matrix(cm, label,
                          normalize=True,
                          title= title +' model confusion matrix. accuracy: '
                                + str(accuracy_score(y_true, y_pred)))
    plt.show()



if __name__ == '__main__':

    ####---- predict the phoneme classes on test set

    pred_gmm_path   = './trainingData/y_pred_gmm.npy'
    pred_jordi_path = './trainingData/y_pred_jordi.npy'
    pred_choi_path  = './trainingData/y_pred_choi.npy'
    pred_dnn_path   = './trainingData/y_pred_dnn.npy'
    true_path       = './trainingData/y_true.npy'

    doClassification()

    ####---- draw the confusion matrix
    draw_confusion_matrix(pred_path=pred_gmm_path,
                          true_path=true_path,
                          title='GMM')
    draw_confusion_matrix(pred_path=pred_jordi_path,
                          true_path=true_path,
                          title='Proposed CNN')
    draw_confusion_matrix(pred_path=pred_choi_path,
                          true_path=true_path,
                          title='Choi CNN')
    draw_confusion_matrix(pred_path=pred_dnn_path,
                          true_path=true_path,
                          title='DNN')



