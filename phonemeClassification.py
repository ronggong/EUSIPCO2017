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

from phonemeMap import dic_pho_label_inv,dic_pho_label
import pickle
import os
import numpy as np



class PhonemeClassification(object):

    def __init__(self):
        self.gmmModel = {}
        self.n = 0
        self.precision = np.double

    def create_gmm(self, model_path):
        """
        load gmmModel
        :return:
        """
        for state in dic_pho_label:
            pkl_file = open(os.path.join(model_path, state + '.pkl'), 'rb')
            self.gmmModel[state] = pickle.load(pkl_file)
            pkl_file.close()
        self.n = len(self.gmmModel)

    def mapb_gmm(self, observations):
        """
        observation probability
        :param observations:
        :return:
        """
        dim_t       = observations.shape[0]
        self.B_map_gmm  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_gmm[dic_pho_label[state],:] = self.gmmModel[state].score_samples(observations)

    def mapb_gmm_getter(self):
        if len(self.B_map_gmm):
            return self.B_map_gmm
        else: return


    def mapb_keras(self, observations, kerasModels_path, jordi=False):
        '''
        dnn observation probability
        :param observations:
        :return:
        '''
        ##-- set environment of the pdnn

        from keras.models import load_model

        model = load_model(kerasModels_path)

        ##-- call pdnn to calculate the observation from the features
        if jordi:
            observations = [observations, observations, observations, observations, observations, observations]

        obs = model.predict_proba(observations, batch_size=128)


        ##-- read the observation from the output

        obs = np.log(obs)

        # print obs.shape, observations.shape

        dim_t       = obs.shape[0]
        self.B_map_keras  = np.zeros((self.n, dim_t), dtype=self.precision)
        # print self.transcription, self.B_map.shape
        for state in dic_pho_label:
            self.B_map_keras[dic_pho_label[state],:] = obs[:, dic_pho_label[state]]


    def mapb_keras_getter(self):
        if len(self.B_map_keras):
            return self.B_map_keras
        else: return




    def prediction(self,obs):
        """
        find index of the max value in axis=1
        :param obs:
        :return:
        """
        y_pred = np.argmax(obs,axis=0)
        return y_pred