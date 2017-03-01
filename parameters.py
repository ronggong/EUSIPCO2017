from os.path import dirname,join

root_path    = join(dirname(__file__))

dataset         = 'qmLonUpfLaosheng'
# dataset         = 'danAll'

am                 = 'cnn'
# am                 = 'gmm'

keras_model_name          = 'choi_danAll_mfccBands_2D_all_optim'

if dataset == 'danAll':
    base_path = 'danAll'
    syllableTierName = 'dian'
elif dataset == 'qmLonUpfLaosheng':
    base_path = 'qmLonUpf/laosheng'
    syllableTierName = 'dian'
phonemeTierName = 'details'

# if you don't have this dataset, please download from http://doi.org/10.5281/zenodo.344932
dataset_path = '/your/path/to/jingju_a_cappella_singing_dataset'

wav_path        = join(dataset_path,'wav',base_path)
textgrid_path   = join(dataset_path,'textgrid',base_path)

gmmModel_path   = join(root_path,'gmmModels',base_path)

scaler_path     = join(root_path, 'pretrainedDLModels', base_path,
                               'scaler_'+dataset+'_phonemeSeg_mfccBands2D.pkl')

kerasModels_jordi_path   = join(root_path, 'pretrainedDLModels', base_path,
                                'keras.cnn_jordi_mfccBands_2D_all_optim.h5')

kerasModels_choi_path   = join(root_path, 'pretrainedDLModels', base_path,
                                'keras.cnn_choi_mfccBands_2D_all_optim.h5')

kerasModels_dnn_path   = join(root_path, 'pretrainedDLModels', base_path,
                                'keras.dnn_optim_mfccBands_neighbor_all.h5')
##-- other parameters

fs = 44100
framesize_t = 0.025     # in second
hopsize_t   = 0.010

framesize   = int(round(framesize_t*fs))
hopsize     = int(round(hopsize_t*fs))

# MFCC params
highFrequencyBound = fs/2 if fs/2<11000 else 11000

varin                = {}
varin['N_feature']   = 40
varin['N_pattern']   = 21                # adjust this param, l in paper

# mfccBands feature half context window length
varin['nlen']        = 10
