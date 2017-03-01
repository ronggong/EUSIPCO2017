# EUSIPCO 2017 Jingju singing voice phoneme classification
The phoneme classification code for EUSIPCO 2017 paper review:
>Timbre Analysis of Music Audio Signals with Convolutional Neural Networks

## Steps for reproducting the experiment results
1. Clone this repository
2. Download Jingju a capella singing dataset from http://doi.org/10.5281/zenodo.344932
3. Change `dataset_path` variable in `parameters.py` to locate the above dataset
4. Install dependencies (see below)
5. Choose `dataset` in `parameters.py` to run experiment on *dan* or *laosheng* dataset
6. Run experiment by `python doPhonemeClassification.py'

## Steps for calculating the mel bands features
1. Execute the steps 1, 2, 3 in **Steps for reproducting the experiment results**
2. Choose `dataset` and `am` variables in `parameters`. Example, `dataset='qmLonUpfLaosheng'` and `am='cnn'` means we would like to extract the *laosheng* features for convolutional neural networks (proposed, Choi models).
3. Run `python phonemeSampleCollection.py` to extract the mel bands features
4. Code for extracting features for MLP model is not included.

## Steps for training proposed, Choi, MLP and GMM models
1. Download pre-computed mel-bands features from http://doi.org/10.5281/zenodo.344935
2. Create a folder named `trainingData` in the root of this repository, then put all '.pickle.gz` feature files into this folder
3. If you don't want to download the pre-computed features, please follow **Steps for calculating the mel bands features**
4. The model training code are located in `pretrainedDLModels` folder. `keras_cnn*` code is for training CNN models (proposed and Choi modes). `keras_dnn*` code is for training MLP model
5. To train GMM models, please set `am='gmm'` in `parameters.py`, then execute steps 1, 2 in **Steps for calculating the mel bands features**

## Dependencies
**Steps for reproducting the experiment results** requires below packages:

`python2 numpy scipy scikit-learn matplotlib essentia`

**Steps for calculating the mel bands features** requires below packages:

`python2 numpy scipy scikit-learn essentia`

**Steps for training proposed, Choi, MLP and GMM models** requires below packages:

`python2 numpy scipy scikit-learn essentia keras theano hyperot`

## License
Affero GNU General Public License version 3
