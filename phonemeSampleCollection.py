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
import pickle,cPickle,gzip

import numpy as np
from sklearn import mixture,preprocessing
from sklearn.model_selection import train_test_split
import essentia.standard as ess

from phonemeMap import *
from parameters import *
from textgridParser import syllableTextgridExtraction
from trainTestSeparation import getRecordingNamesSimi
from Fdeltas import Fdeltas
from Fprev_sub import Fprev_sub


winAnalysis     = 'hann'
N               = 2 * framesize                     # padding 1 time framesize
SPECTRUM        = ess.Spectrum(size=N)
MFCC80            = ess.MFCC(sampleRate         =fs,
                           highFrequencyBound   =highFrequencyBound,
                           inputSize            =framesize+1,
                           numberBands          =80)

# this MFCC is for pattern classification, which numberBands always be by default
MFCC40          = ess.MFCC(sampleRate           =fs,
                           highFrequencyBound   =highFrequencyBound,
                           inputSize            =framesize+1)
WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N-framesize)

def getFeature(audio, d=True, nbf=False):

    '''
    MFCC of give audio interval [p[0],p[1]]
    :param audio:
    :param p:
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        # mfccFrame       = mfccFrame[1:]
        mfcc.append(mfccFrame)

    if d:
        mfcc            = np.array(mfcc).transpose()
        dmfcc           = Fdeltas(mfcc,w=5)
        ddmfcc          = Fdeltas(dmfcc,w=5)
        feature         = np.transpose(np.vstack((mfcc,dmfcc,ddmfcc)))
    else:
        feature         = np.array(mfcc)

    if not d and nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for w_r in range(1,6):
            mfcc_right_shifted = Fprev_sub(mfcc, w=w_r)
            mfcc_left_shifted = Fprev_sub(mfcc, w=-w_r)
            mfcc_out = np.vstack((mfcc_out, mfcc_left_shifted, mfcc_right_shifted))
        feature = np.array(np.transpose(mfcc_out),dtype='float32')

    # print feature.shape

    return feature

def getMFCCBands1D(audio, nbf=False):

    '''
    mel bands feature [p[0],p[1]], this function only for pdnn acoustic model training
    output feature is a 1d vector
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_right_shifted_1 = Fprev_sub(mfcc, w=1)
        mfcc_left_shifted_1 = Fprev_sub(mfcc, w=-1)
        mfcc_right_shifted_2 = Fprev_sub(mfcc, w=2)
        mfcc_left_shifted_2 = Fprev_sub(mfcc, w=-2)
        feature = np.transpose(np.vstack((mfcc,
                                          mfcc_right_shifted_1,
                                          mfcc_left_shifted_1,
                                          mfcc_right_shifted_2,
                                          mfcc_left_shifted_2)))
    else:
        feature = mfcc

    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature

def getMFCCBands2D(audio, nbf=False, nlen=10):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for ii in range(1,nlen+1):
            mfcc_right_shift    = Fprev_sub(mfcc, w=ii)
            mfcc_left_shift     = Fprev_sub(mfcc, w=-ii)
            mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
        feature = mfcc_out.transpose()
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature

def getMBE(audio):
    '''
    mel band energy feature
    :param audio:
    :return:
    '''

    mfccBands = []
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):

        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC40(mXFrame)
        mfccBands.append(bands)
    feature         = np.array(mfccBands)
    return feature

def featureLabel(dic_pho_feature_train):
    '''
    organize the training feature and label
    :param dic_pho_feature_train: input dictionary, key: phoneme, value: feature vectors
    :return:
    '''
    feature_all = []
    label_all   = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all,dtype='int64')

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_all)
    feature_all = scaler.transform(feature_all)

    return feature_all, label_all, scaler

def featureReshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')

    for ii in range(n_sample):
        print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped

def dumpFeature(recordings,syllableTierName,phonemeTierName,feature_type='mfcc',dmfcc=True,nbf=False):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    ##-- dictionary feature
    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)

        # audio
        wav_full_filename   = os.path.join(wav_path,recording+'.wav')
        audio               = ess.MonoLoader(downmix = 'left', filename = wav_full_filename, sampleRate = fs)()

        if feature_type == 'mfcc':
            # MFCC feature
            mfcc = getFeature(audio, d=dmfcc, nbf=nbf)
        elif feature_type == 'mfccBands1D':
            mfcc = getMFCCBands1D(audio, nbf=nbf)
            mfcc = np.log(100000*mfcc+1)
        elif feature_type == 'mfccBands2D':
            mfcc = getMFCCBands2D(audio, nbf=nbf, nlen=varin['nlen'])
            mfcc = np.log(100000*mfcc+1)
        else:
            print(feature_type+' is not exist.')
            raise

        for ii,pho in enumerate(nestedPhonemeLists):
            print 'calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                key = dic_pho_map[p[2]]

                sf = round(p[0]*fs/float(hopsize)) # starting frame
                ef = round(p[1]*fs/float(hopsize)) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = mfcc_p
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],mfcc_p))

    return dic_pho_feature

def bicGMMModelSelection(X):
    '''
    bic model selection
    :param X: features - observation * dimension
    :return:
    '''
    lowest_bic = np.infty
    bic = []
    n_components_range  = [10,15,20,25,30,35,40,45,50,55,60,65,70]
    best_n_components   = n_components_range[0]
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        print 'Fitting GMM with n_components =',str(n_components)
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='diag')
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_n_components = n_components
            best_gmm          = gmm

    return best_n_components,gmm

def modelSelection(featureFilename):
    '''
    print the best n_component for the phoneme in feature file
    :param featureFilename:
    :return:
    '''
    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_pho_feature_train = pickle.load(pkl_file)
    pkl_file.close()

    print len(dic_pho_feature_train.keys())
    for ii,key in enumerate(dic_pho_feature_train):
        X = dic_pho_feature_train[key]
        best_n_components = bicGMMModelSelection(X)

        print 'The best n_components for',key,'is',str(best_n_components)

def processAcousticModelTrain(mode,syllableTierName,phonemeTierName,featureFilename,gmmModel_path):
    '''

    :param mode: sourceSeparation, qmLonUpfLaosheng
    :param syllableTierName: 'pinyin', 'dian'
    :param phonemeTierName: 'details'
    :param featureFilename: 'dic_pho_feature_train.pkl'
    :param gmmModel_path: in parameters.py
    :return:
    '''

    # model training
    dic_pho_feature_train = dumpFeature(recordings_train,syllableTierName,phonemeTierName)

    output = open(featureFilename, 'wb')
    pickle.dump(dic_pho_feature_train, output)
    output.close()

    # model loading
    pkl_file = open(featureFilename, 'rb')
    dic_pho_feature_train = pickle.load(pkl_file)
    pkl_file.close()

    g = mixture.GaussianMixture(n_components=40,covariance_type='diag')

    print len(dic_pho_feature_train.keys())
    for ii,key in enumerate(dic_pho_feature_train):
        # print key, dic_pho_feature_train[key].shape

        print 'fitting gmm ', key, ' ', str(ii), ' of ', str(len(dic_pho_feature_train.keys()))

        ##-- try just fit the first dim of MFCC
        # x = np.expand_dims(dic_pho_feature_train[key][:,0],axis=1)

        x = dic_pho_feature_train[key]
        print x.shape
        g.fit(x)

        output = open(os.path.join(gmmModel_path,key+'.pkl'),'wb')
        pickle.dump(g, output)
        output.close()

if __name__ == '__main__':

    if am == 'gmm':
        # dump GMM acoustic models
        processAcousticModelTrain(mode=dataset,
                                  syllableTierName=syllableTierName,
                                  phonemeTierName=phonemeTierName,
                                  featureFilename='./trainingData/dic_pho_feature_train_'+dataset+'.pkl',
                                  gmmModel_path=gmmModel_path)

    elif am == 'cnn':
        # dump feature for CNN acoustic model training
        recordings_train = getRecordingNamesSimi('TRAIN', dataset)

        dic_pho_feature_train = dumpFeature(recordings_train,
                                            syllableTierName,
                                            phonemeTierName,
                                            feature_type='mfccBands2D',
                                            dmfcc=False,
                                            nbf=True)

        if dataset == 'qmLonUpfLaosheng':
            # dump feature dan role-type, dan dataset is bigger so we split it to two parts
            feature_all,label_all, scaler = featureLabel(dic_pho_feature_train)

            pickle.dump(scaler,open('./pretrainedDLModels/qmLonUpf/laosheng/scaler_'+dataset+'_phonemeSeg_mfccBands2D.pkl', 'wb'))

            feature_all = featureReshape(feature_all,nlen=varin['nlen'])

            cPickle.dump((feature_all, label_all),
                         gzip.open('./trainingData/train_set_all_'+dataset+'_phonemeSeg_mfccBands2D.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)

            feature_train, feature_validation, label_train, label_validation = \
                train_test_split(feature_all, label_all, test_size=0.2, stratify=label_all)

            #-- dump feature vectors training and validation sets separately
            cPickle.dump((feature_train, label_train),
                         gzip.open('./trainingData/train_set_'+dataset+'_phonemeSeg_mfccBands2D.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)
            cPickle.dump((feature_validation, label_validation),
                         gzip.open('./trainingData/validation_set_'+dataset+'_phonemeSeg_mfccBands2D.pickle.gz', 'wb'), cPickle.HIGHEST_PROTOCOL)

            print feature_train.shape, len(feature_validation), len(label_train), len(label_validation)

        elif dataset == 'danAll':
            # dump feature dan role-type, dan dataset is bigger so we split it to two parts
            feature_all,label_all, scaler = featureLabel(dic_pho_feature_train)

            pickle.dump(scaler, open('./pretrainedDLModels/danAll/scaler_' + dataset + '_phonemeSeg_mfccBands2D.pkl', 'wb'))

            feature_all = featureReshape(feature_all, nlen=varin['nlen'])

            # split feature all
            n_sample_all = int(feature_all.shape[0] / 3)

            cPickle.dump((feature_all[:n_sample_all, :, :], label_all[:n_sample_all]),
                         gzip.open('./trainingData/train_set_all_' + dataset + '_phonemeSeg_mfccBands2D_part0.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            cPickle.dump((feature_all[n_sample_all:2 * n_sample_all, :, :], label_all[n_sample_all:2 * n_sample_all]),
                         gzip.open('./trainingData/train_set_all_' + dataset + '_phonemeSeg_mfccBands2D_part1.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            cPickle.dump((feature_all[2 * n_sample_all:, :, :], label_all[2 * n_sample_all:]),
                         gzip.open('./trainingData/train_set_all_' + dataset + '_phonemeSeg_mfccBands2D_part2.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            feature_train, feature_validation, label_train, label_validation = \
                train_test_split(feature_all, label_all, test_size=0.2, stratify=label_all)

            # -- dump feature vectors training and validation sets separately
            n_sample_train = int(feature_train.shape[0]/2)

            cPickle.dump((feature_train[:n_sample_train,:,:], label_train[:n_sample_train]),
                         gzip.open('./trainingData/train_set_' + dataset + '_phonemeSeg_mfccBands2D_part0.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)
            cPickle.dump((feature_train[n_sample_train:,:,:], label_train[n_sample_train:]),
                         gzip.open('./trainingData/train_set_' + dataset + '_phonemeSeg_mfccBands2D_part1.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            cPickle.dump((feature_validation, label_validation),
                         gzip.open('./trainingData/validation_set_' + dataset + '_phonemeSeg_mfccBands2D.pickle.gz', 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            print feature_train.shape, len(label_train), len(label_validation)