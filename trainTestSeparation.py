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
from itertools import combinations

import numpy as np
import json
import operator

from textgridParser import syllableTextgridExtraction
from parameters import *
from pinyinMap import dic_pinyin_2_initial_final_map
from phonemeMap import *

def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                    recordings.append(file_prefix)

    return recordings

def testRecordings(boundaries,proportion_testset):
    '''
    :param boundaries: a list of boundary number of each recording
    :param proportion_testset:
    :return: a list of test recordings
    '''

    sum_boundaries = sum(boundaries)
    boundaries     = np.array(boundaries)
    subsets        = []

    for ii in range(1,len(boundaries)):
        for subset in combinations(range(len(boundaries)),ii):
            subsets.append([subset,abs(sum(boundaries[np.array(subset)])/float(sum_boundaries)-proportion_testset)])

    subsets        = np.array(subsets)
    subsets_sorted = subsets[np.argsort(subsets[:,1]),0]

    return subsets_sorted[0]

def getRecordingNumber(train_test_string,mode):
    '''
    return number of recording, test recording contains 25% ground truth boundaries
    :param train_test_string: 'TRAIN' or 'TEST'
    :return:
    '''
    if mode == 'sourceSeparation':
        train_recordings        = [0,2,4,5,6,8,9,10,12,13,14]
        test_recordings         = [1,3,7,11]
    elif mode == 'qmLonUpfLaosheng':
        train_recordings        = [0,1,3,4,5,6,9,10,11,13,14,15,16,21,22]
        test_recordings         = [2, 7, 8, 12, 17, 18, 19, 20]
    else:
        pass

    if train_test_string == 'TRAIN':
        number_recording     = train_recordings
    else:
        number_recording     = test_recordings

    return number_recording

def getRecordingNames(train_test_string,mode):

    if mode == 'qmLonUpfLaosheng':
        recordings_train = ['lseh-Tan_Yang_jia-Hong_yang_dong-qm', 'lseh-Wei_guo_jia-Hong_yang_dong01-lon', 'lseh-Wei_guo_jia-Hong_yang_dong02-qm', 'lseh-Yi_lun_ming-Wen_zhao_guan-qm', 'lseh-Zi_na_ri-Hong_yang_dong-qm', 'lsfxp-Yang_si_lang-Si_lang_tan_mu-lon', 'lsxp-Huai_nan_wang-Huai_he_ying01-lon', 'lsxp-Jiang_shen_er-San_jia_dian02-qm', 'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm', 'lsxp-Quan_qian_sui-Gan_lu_si-qm', 'lsxp-Shen_gong_wu-Gan_lu_si-qm', 'lsxp-Shi_ye_shuo-Ding_jun_shan-qm', 'lsxp-Wo_ben_shi-Kong_cheng_ji-qm', 'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf', 'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm', 'lsxp-Xi_ri_you-Zhu_lian_zhai-qm']
        recordings_test  = ['lseh-Wo_ben_shi-Qiong_lin_yan-qm', 'lsxp-Guo_liao_yi-Wen_zhao_guan-qm', 'lsxp-Guo_liao_yi-Wen_zhao_guan-upf', 'lsxp-Huai_nan_wang-Huai_he_ying02-qm', 'lsxp-Jiang_shen_er-San_jia_dian01-1-upf', 'lsxp-Jiang_shen_er-San_jia_dian01-2-upf', 'lsxp-Wo_zheng_zai-Kong_cheng_ji02-qm']
    elif mode == 'sourceSeparation':
        recordings_train = ['shiwenhui_tingxiongyan', 'wangjiangting_zhijianta', 'xixiangji_diyilai', 'xixiangji_luanchouduo', 'xixiangji_manmufeng', 'xixiangji_zhenmeijiu', 'yutangchun_yutangchun', 'zhuangyuanmei_daocishi', 'zhuangyuanmei_tianbofu', 'zhuangyuanmei_zhenzhushan', 'zhuangyuanmei_zinari']
        recordings_test  = ['wangjiangting_dushoukong', 'xixiangji_biyuntian', 'xixiangji_xianzhishuo', 'zhuangyuanmei_fudingkui']
    elif mode == 'danAll':
        recordings_train = ['daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon', 'daeh-Yi_sha_shi-Suo_lin_nang-qm',
                            'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon', 'danbz-Bei_jiu_chan-Chun_gui_men01-qm',
                            'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm', 'danbz-Kan_dai_wang-Ba_wang_bie_ji03-qm',
                            'danbz-Qing_chen_qi-Shi_yu_zhuo-qm', 'daspd-Du_shou_kong-Wang_jiang_ting-upf',
                            'daxp-Chun_qiu_ting-Suo_lin_nang01-qm', 'daxp-Chun_qiu_ting-Suo_lin_nang03-qm',
                            'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon', 'daxp-Jiao_Zhang_sheng-Hong_niang01-qm',
                            'daxp-Jiao_Zhang_sheng-Hong_niang05-qm', 'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf',
                            'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm','daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai06-qm',
                            'daxp-Zhe_cai_shi-Suo_lin_nang01-qm', 'shiwenhui_tingxiongyan', 'wangjiangting_dushoukong',
                            'wangjiangting_zhijianta', 'xixiangji_biyuntian','xixiangji_manmufeng', 'xixiangji_xianzhishuo',
                            'xixiangji_zhenmeijiu', 'yutangchun_yutangchun', 'zhuangyuanmei_daocishi',
                            'zhuangyuanmei_fudingkui', 'zhuangyuanmei_zhenzhushan']

        recordings_test  = ['daeh-Bie_yuan_zhong-Mei_fei-qm', 'daeh-You_He_hou-He_hou_ma_dian-qm',
                            'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm', 'dafeh-Mo_lai_you-Liu_yue_xue-qm',
                            'danbz-Bei_jiu_chan-Chun_gui_men03-qm', 'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon',
                            'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm', 'daxp-Jiao_Zhang_sheng-Hong_niang04-qm',
                            'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm', 'daxp-Xi_ri_li-Gan_lu_si-qm',
                            'xixiangji_diyilai', 'xixiangji_luanchouduo','zhuangyuanmei_tianbofu', 'zhuangyuanmei_zinari']

    if train_test_string == 'TRAIN':
        name_recordings      = recordings_train
    else:
        name_recordings      = recordings_test
    return name_recordings

def getRecordingNamesSimi(train_test,mode):
    '''
    return recordings names, test recordings in score dataset
    :param mode: role type
    :return:
    '''
    if mode == 'qmLonUpfLaosheng':
        recordings_train = ['lseh-Tan_Yang_jia-Hong_yang_dong-qm',
                            'lsfxp-Yang_si_lang-Si_lang_tan_mu-lon',
                            'lsxp-Qian_bai_wan-Si_lang_tang_mu01-qm',
                            'lsxp-Quan_qian_sui-Gan_lu_si-qm',
                            'lsxp-Shi_ye_shuo-Ding_jun_shan-qm',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji04-qm',
                            'lsxp-Xi_ri_you-Zhu_lian_zhai-qm',
                            'lseh-Wo_ben_shi-Qiong_lin_yan-qm',
                            'lsxp-Guo_liao_yi-Wen_zhao_guan-qm',
                            'lsxp-Guo_liao_yi-Wen_zhao_guan-upf',
                            'lsxp-Huai_nan_wang-Huai_he_ying01-lon',
                            'lsxp-Wo_zheng_zai-Kong_cheng_ji02-qm']

        # these arias are in the score corpus 88 lines
        recordings_test = [
                      'lseh-Wei_guo_jia-Hong_yang_dong01-lon',
                      'lseh-Wei_guo_jia-Hong_yang_dong02-qm',
                      'lseh-Yi_lun_ming-Wen_zhao_guan-qm',
                      'lseh-Zi_na_ri-Hong_yang_dong-qm', # 4,5 not in corpus
                      'lsxp-Huai_nan_wang-Huai_he_ying02-qm', # 0,1,2,3 not in corpus
                      'lsxp-Jiang_shen_er-San_jia_dian01-1-upf',
                      'lsxp-Jiang_shen_er-San_jia_dian01-2-upf',
                      'lsxp-Jiang_shen_er-San_jia_dian02-qm',
                      'lsxp-Shen_gong_wu-Gan_lu_si-qm',
                      'lsxp-Wo_ben_shi-Kong_cheng_ji-qm']
    elif mode == 'danAll':

        recordings_train = ['daspd-Du_shou_kong-Wang_jiang_ting-upf',
                            'daeh-Bie_yuan_zhong-Mei_fei-qm',
                            'daeh-Yang_Yu_huan-Tai_zhen_wai_zhuan-lon',
                            'daeh-Yi_sha_shi-Suo_lin_nang-qm',
                            'daeh-You_He_hou-He_hou_ma_dian-qm',
                            'dafeh-Bi_yun_tian-Xi_xiang_ji01-qm',
                            'dafeh-Mo_lai_you-Liu_yue_xue-qm',
                            'dagbz-Feng_xiao_xiao-Yang_men_nv_jiang-lon',
                            'danbz-Bei_jiu_chan-Chun_gui_men03-qm',
                            'danbz-Kan_dai_wang-Ba_wang_bie_ji03-qm',
                            'danbz-Qing_chen_qi-Shi_yu_zhuo-qm',
                            'daspd-Hai_dao_bing-Gui_fei_zui_jiu01-lon',
                            'daxp-Jiao_Zhang_sheng-Hong_niang04-qm',
                            'daxp-Jiao_Zhang_sheng-Hong_niang05-qm',
                            'daxp-Chun_qiu_ting-Suo_lin_nang03-qm',
                            'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai02-qm',
                            'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai04-qm',
                            'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai06-qm',
                            'shiwenhui_tingxiongyan',
                            'xixiangji_biyuntian',
                            'xixiangji_diyilai',
                            'xixiangji_luanchouduo',
                            'xixiangji_manmufeng',
                            'xixiangji_xianzhishuo',
                            'xixiangji_zhenmeijiu',
                            'yutangchun_yutangchun',
                            'zhuangyuanmei_daocishi',
                            'zhuangyuanmei_fudingkui',
                            'zhuangyuanmei_tianbofu',
                            'zhuangyuanmei_zhenzhushan']

        # 72 lines (estimated) in score corpus

        recordings_test = ['zhuangyuanmei_zinari',
                           'daxp-Chun_qiu_ting-Suo_lin_nang01-qm',
                           'daxp-Meng_ting_de-Mu_Gui_ying_gua_shuai01-upf',
                           'daxp-Guan_Shi_yin-Tian_nv_san_hua-lon',
                           'danbz-Bei_jiu_chan-Chun_gui_men01-qm',
                           'danbz-Kan_dai_wang-Ba_wang_bie_ji01-qm',
                           'daspd-Hai_dao_bing-Gui_fei_zui_jiu02-qm',
                           'daxp-Jiao_Zhang_sheng-Hong_niang01-qm',
                           'daxp-Zhe_cai_shi-Suo_lin_nang01-qm',
                           'daxp-Xi_ri_li-Gan_lu_si-qm',
                           'wangjiangting_zhijianta',
                           'wangjiangting_dushoukong'
                           ]

    if train_test == 'TRAIN':
        return recordings_train
    else:
        return recordings_test

def transHelper(pii,piii):
    if pii in vowels:
        p0 = 'vowels'
    elif pii in semivowels:
        p0 = 'semivowels'
    elif pii in diphtongs:
        p0 = 'diphtongs'
    elif pii in compoundfinals:
        p0 = 'compoundfinals'
    elif pii in nonvoicedconsonants:
        p0 = 'nonvoicedconsonants'
    elif pii in voicedconsonants:
        p0 = 'voicedconsonants'
    elif pii in silornament:
        p0 = 'silornament'

    if piii in vowels:
        p1 = 'vowels'
    elif piii in semivowels:
        p1 = 'semivowels'
    elif piii in diphtongs:
        p1 = 'diphtongs'
    elif piii in compoundfinals:
        p1 = 'compoundfinals'
    elif piii in nonvoicedconsonants:
        p1 = 'nonvoicedconsonants'
    elif piii in voicedconsonants:
        p1 = 'voicedconsonants'
    elif piii in silornament:
        p1 = 'silornament'

    return p0,p1

def textgridStat(textgrid_path,syllableTierName,phonemeTierName):
    '''
    syllableTierName: pinyin or dian
    phonemeTierName: details
    '''

    recordings = getRecordings(textgrid_path)
    # recordings = getRecordingNames('TEST',dataset)

    numLine_all, numSyllable_all, numVoiced_all, numUnvoiced_all = 0,0,0,0
    lengthLine_all, lengthSyllable_all, lengthVoiced_all, lengthUnvoiced_all = [],[],[],[]
    numVowels, numSemivowels, numDiphtongs, numCompoundfinals, \
    numNonvoicedconsonants, numVoicedconsonants, numSilornament = 0,0,0,0,0,0,0

    # from * transit to phoneme
    trans2n     = []
    trans2i     = []
    trans2N     = []
    trans2u     = []

    dict_numTrans_phoneme = {}
    for tp in trans_phoneme:
        dict_numTrans_phoneme[tp] = 0

    for recording in recordings:
        print 'processing recording:',recording
        nestedSyllableLists, numLines, numSyllables \
            = syllableTextgridExtraction(textgrid_path,recording,'line',syllableTierName)
        numLine_all += numLines
        for line in nestedSyllableLists:
            lengthLine_all.append(line[0][1]-line[0][0])


        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)
        numSyllable_all += numSyllables
        for pho in nestedPhonemeLists:
            lengthSyllable_all.append(pho[0][1]-pho[0][0])
            for p in pho[1]:
                if p[2] in ['c','k','f','x']:
                    numUnvoiced_all += 1
                    lengthUnvoiced_all.append(p[1]-p[0])
                else:
                    numVoiced_all   += 1
                    lengthVoiced_all.append(p[1]-p[0])
            for p in pho[1]:
                if p[2] in vowels:
                    numVowels += 1
                elif p[2] in semivowels:
                    numSemivowels += 1
                elif p[2] in diphtongs:
                    numDiphtongs += 1
                elif p[2] in compoundfinals:
                    numCompoundfinals += 1
                elif p[2] in nonvoicedconsonants:
                    numNonvoicedconsonants += 1
                elif p[2] in voicedconsonants:
                    numVoicedconsonants += 1
                elif p[2] in silornament:
                    numSilornament += 1

        # transition
        for pho in nestedPhonemeLists:

            for ii in range(len(pho[1])-1):
                p0,p1 = transHelper(pho[1][ii][2],pho[1][ii+1][2])

                for tp in trans_phoneme:
                    if p0 == tp.split('_')[0] and p1 == tp.split('_')[1]:
                        dict_numTrans_phoneme[tp] += 1

                if pho[1][ii+1][2] == 'n':
                    trans2n.append(pho[1][ii][2]+'_'+pho[1][ii+1][2])
                elif pho[1][ii+1][2] == 'i':
                    trans2i.append(pho[1][ii][2]+'_'+pho[1][ii+1][2])
                elif pho[1][ii+1][2] == 'N':
                    trans2N.append(pho[1][ii][2]+'_'+pho[1][ii+1][2])
                elif pho[1][ii+1][2] == 'u':
                    trans2u.append(pho[1][ii][2]+'_'+pho[1][ii+1][2])


    occurrence_threshold = sum(dict_numTrans_phoneme.values())*0.005
    sorted_numTrans_phoneme = sorted(dict_numTrans_phoneme.items(), key=operator.itemgetter(1))[::-1]

    sorted_numTrans_phoneme_threshed = []
    for sntp in sorted_numTrans_phoneme:
        if sntp[1]>occurrence_threshold:
            sorted_numTrans_phoneme_threshed.append(sntp)

    ##-- output statistics of the dataset
    print 'num recordings %i' % len(recordings)
    print 'num lines %i, num syllables %i, voiced phonemes %i, unvoiced phonemes %i' % (numLine_all, numSyllable_all, numVoiced_all, numUnvoiced_all)
    print 'avg len (s) lines %.3f, syllables %.3f, voiced phonemes %.3f, unvoiced phonemes %.3f' % (np.mean(lengthLine_all), np.mean(lengthSyllable_all), np.mean(lengthVoiced_all), np.mean(lengthUnvoiced_all))
    print 'std len (s) lines %.3f, syllables %.3f, voiced phonemes %.3f, unvoiced phonemes %.3f' % (np.std(lengthLine_all), np.std(lengthSyllable_all), np.std(lengthVoiced_all), np.std(lengthUnvoiced_all))
    print 'numVowels %d, numSemivowels %d, numDiphtongs %d, numCompoundfinals %d, numNonvoicedconsonants %d, numVoicedconsonants %d, numSilornament %d' % (numVowels, numSemivowels, numDiphtongs, numCompoundfinals, numNonvoicedconsonants, numVoicedconsonants, numSilornament)
    print sorted_numTrans_phoneme_threshed

    print set(trans2n)
    print set(trans2i)
    print set(trans2N)
    print set(trans2u)

def getCorrectTransCategories(transPhoneme):
    '''
    :param correctTransPhonemeAll:
    :return: dict of phoneme transitions
    '''
    dict_numTrans_phoneme = {}
    for tp in trans_phoneme:
        dict_numTrans_phoneme[tp] = 0

    for correctTP in transPhoneme:
        if len(correctTP):
            cp0 = correctTP[0]
            cp1 = correctTP[1]
            p0,p1 = transHelper(cp0,cp1)

            for tp in trans_phoneme:
                if p0 == tp[:len(p0)] and p1 == tp[-len(p1):]:
                    dict_numTrans_phoneme[tp] += 1
    sorted_numTrans_phoneme = sorted(dict_numTrans_phoneme.items(), key=operator.itemgetter(1))[::-1]
    return sorted_numTrans_phoneme


def getValidTransGt(textgrid_path,syllableTierName,phonemeTierName):
    recordings = getRecordingNames('TEST',dataset)
    numValidTrans = 0
    for recording in recordings:
        print 'get valid trans gt processing recording:',recording
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)
        for pho in nestedPhonemeLists:
            if pho[1][0][2] in ['m','l','n','c','f','k','s','x',"r\\'",'w','j']:
                numValidTrans += 1
            for ii in range(len(pho[1])-1):
                if pho[1][ii][2]+'_'+pho[1][ii+1][2] in tails_comb_i+tails_comb_N+tails_comb_n+tails_comb_u:
                    numValidTrans += 1
    return numValidTrans

def validTransStat(detectedTransPhonemeAll,correctTransPhonemeAll):
    '''
    only count the valid transition boundaries: valid initials, valid finals
    :param detectedTransPhonemeAll:
    :param correctTransPhonemeAll:
    :return:
    '''

    dict_numTrans_phoneme = {}
    for tp in trans_phoneme:
        dict_numTrans_phoneme[tp] = 0

    numValidTransCorrect    = 0
    numValidTransDetected   = 0

    for detectedTP in detectedTransPhonemeAll:
        if detectedTP[1]+'_'+detectedTP[2] in tails_comb_i_map+tails_comb_n_map+tails_comb_N_map+tails_comb_u_map:
            numValidTransDetected += 1
        if detectedTP[1] in ['nvc','vc',"r\\'",'u','j'] and detectedTP[0] == 0:
            numValidTransDetected += 1

    for correctTP in correctTransPhonemeAll:
        if correctTP[1]+'_'+correctTP[2] in tails_comb_i+tails_comb_N+tails_comb_n+tails_comb_u:
            numValidTransCorrect += 1
        if correctTP[1] in ['m','l','n','c','f','k','s','x',"r\\'",'w','j'] and correctTP[0] == 0:
            numValidTransCorrect += 1

        # if len(correctTP):
        #     cp0 = correctTP[1]
        #     cp1 = correctTP[2]
        #     p0,p1   = transHelper(cp0,cp1)

            # for tp in trans_phoneme:
            #     if p0 == tp.split('_')[0] and p1 == tp.split('_')[1]:
            #         dict_numTrans_phoneme[tp] += 1

    # sorted_numTrans_phoneme = sorted(dict_numTrans_phoneme.items(), key=operator.itemgetter(1))[::-1]
    # json.dump(sorted_numTrans_phoneme, open("dict_numTrans_phoneme_"+dataset+".json",'w'))

    numValidTransGt = getValidTransGt(textgrid_path,syllableTierName,phonemeTierName)

    return numValidTransDetected,numValidTransGt,numValidTransCorrect

    # json.dump(sorted_numTrans_phoneme_threshed, open("../dict_numTrans_phoneme_"+dataset+"_gt.json",'w'))

def validPhoDecoded(pho_decoded,time_boundary_decoded):
    '''
    This function finds the valid boundaries and merges the decoded phonemes
    :param pho_decoded: decoded phonemes
    :param time_boundary_decoded: decoded time boundaries
    :return:
    '''

    valid_pho_decoded = []
    valid_time_boundary_decoded = []

    if len(pho_decoded)==1:
        valid_pho_decoded=pho_decoded
        valid_time_boundary_decoded=time_boundary_decoded

    ##-- replace the non valid phonemes to *
    elif len(pho_decoded)>1:

        # intial phoneme
        if pho_decoded[0] in ['nvc','vc',"r\\'",'u','j']:
            valid_pho_decoded.append(pho_decoded[0])
            valid_time_boundary_decoded.append(time_boundary_decoded[0])
        else:
            valid_pho_decoded.append('*')

        # tail phoneme, check the boundary transition, if not in, replace the tail to *
        for ii_pho in range(len(pho_decoded)-1):
            if pho_decoded[ii_pho]+'_'+pho_decoded[ii_pho+1] in tails_comb_u_map+tails_comb_N_map+tails_comb_n_map+tails_comb_i_map:
                valid_pho_decoded.append(pho_decoded[ii_pho+1])

                if ii_pho == 0 and pho_decoded[ii_pho] in ['nvc','vc',"r\\'",'u','j']:
                    pass
                else:
                    valid_time_boundary_decoded.append(time_boundary_decoded[ii_pho])
            else:
                valid_pho_decoded.append('*')

        ##-- indices combined of *
        idx_combine = []
        idx_combine_element = []
        for ii_vpd in range(len(valid_pho_decoded)):
            if valid_pho_decoded[ii_vpd] == '*':
                idx_combine_element.append(ii_vpd)
            else:
                if len(idx_combine_element):
                    idx_combine.append(idx_combine_element)
                    idx_combine_element = []
                idx_combine.append([ii_vpd])

            if ii_vpd == len(valid_pho_decoded)-1 and len(idx_combine_element):
                idx_combine.append(idx_combine_element)

        ##-- merge by concatenating the * phonemes
        valid_pho_decoded = []
        for cb in idx_combine:
            if len(cb)>1:
                pho_decoded_element = [pho_decoded[ii_cb] for ii_cb in cb]
                vpd_element         = '_'.join(pho_decoded_element)
            else:
                vpd_element = pho_decoded[cb[0]]
            valid_pho_decoded.append(vpd_element)

        ##-- deal with the problem if pho_decoded longer than time_boundary
        if len(valid_pho_decoded)-1 > len(valid_time_boundary_decoded):
            valid_pho_decoded[len(valid_time_boundary_decoded)] = '_'.join(valid_pho_decoded[len(valid_time_boundary_decoded):])
            for ii_vpd in valid_pho_decoded[len(valid_time_boundary_decoded)+1:]:
                valid_pho_decoded.pop(-1)

    return valid_pho_decoded,valid_time_boundary_decoded

def textgridError(textgrid_path,syllableTierName,phonemeTierName):
    '''
    find annotation errors: phoneme not in the dic_pho_map keys list
    '''

    recordings = getRecordings(textgrid_path)
    error = []
    for recording in recordings:
        print 'processing recording:',recording
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)
        for pho in nestedPhonemeLists:
            if len(pho[1]) > 5 or pho[0][2] not in dic_pinyin_2_initial_final_map.keys():
                errorInfo = (recording,str(pho[0][0]),pho[0][2],str([p[2] for p in pho[1]]))
                error.append(errorInfo)
            for p in pho[1]:
                ##-- for debug the Textgrid phoneme annotation
                if p[2] not in dic_pho_map.keys():
                    errorInfo = (recording,str(p[0]),p[2])
                    error.append(errorInfo)
    with open('textgridError.txt','wb') as f:
        for errorInfo in error:
            f.write(' '.join(errorInfo))
            f.write('\n')
            f.write('\n')

def findTestRecordingNumber(textgrid_path,syllableTierName,phonemeTierName):
    '''
    find test recording numbers
    '''
    recordings = getRecordings(textgrid_path)
    boundaries  = []
    for recording in recordings:
        print 'processing recording:',recording
        boundaries_oneSong  = 0
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,recording,syllableTierName,phonemeTierName)
        for nestedPho in nestedPhonemeLists:
            boundaries_oneSong  += len(nestedPho[1])-1
        boundaries.append(boundaries_oneSong)
    proportion_testset = 0.25
    print 'processing boundary ...'
    print boundaries
    index_testset      = testRecordings(boundaries, proportion_testset)
    # output test set index
    return index_testset

if __name__ == '__main__':

    textgridStat(textgrid_path=textgrid_path,syllableTierName=syllableTierName,phonemeTierName=phonemeTierName)
    # textgridError(textgrid_path=textgrid_path,syllableTierName='dian',phonemeTierName='details')

    # recordings = getRecordings(textgrid_path)
    # print len(recordings)
    # number_train = getRecordingNumber('TRAIN',mode='sourceSeparation')
    # number_test  = getRecordingNumber('TEST',mode='sourceSeparation')

    # recordings_train = getRecordingNames('TRAIN',mode=dataset)
    # recordings_test = getRecordingNames('TEST',mode=dataset)
    #
    # print len(recordings_train),len(recordings_test)

    # split recordings into train and test set
    # number_test = findTestRecordingNumber(textgrid_path=textgrid_path,syllableTierName=syllableTierName,phonemeTierName=phonemeTierName)
    # number_train = [ii for ii in xrange(len(recordings)) if ii not in number_test]
    # recordings_train = [recordings[ii] for ii in number_train]
    # recordings_test  = [recordings[ii] for ii in number_test]
    # print number_test
    # print recordings_train
    # print recordings_test

