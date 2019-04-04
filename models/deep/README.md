# A source code implementing Deep CNN for MedleyDB

This repository contains code for musical instrument recognition experiments for the deep learning part of the master thesis. 
We provide the code for data preprocessing, training and evaluation of our approach. The code needs to be executed in the same order as we discuss them below. 

## Dataset

We use MedleyDB dataset, which can be downloaded from [MTG website] (www.medleydb.weebly.com/downloads.html) and change the paths to the training and testing splits at the settings file: `../../settings.txt`

## Preprocessing

Before to run the preprocessing script, please, create specific folder for every model which you would like to preprocess the data for. 
The following notation is using (consult `../../settings.txt`): `$MEDLEY_TRAIN_FEATURE_BASEPATH/model_name` and `$MEDLEY_TEST_FEATURE_BASEPATH/model_name`, where `model_name` is referenced to one of the model filenames in `./experiments/models` folder.
Usage:

```bash
python preprocessing_train.py -m model_name -c channel_name -t dataset_type -l window_configuration
python preprocessing_test.py -m model_name -c channel_name -t dataset_type -l window_configuration

```

Currently, we support only the model `han16`  which is referenced as han16 in the thesis. The dataset_types are : {original, harmonic and residual}. The channel_names are : {left, right, mid, side}. The window_configurations are : {window_size}_h{percentage_hop} for which datasets have already been extracted.

For example, 
```bash
python preprocessing_train.py -m han16 -c left -t harmonic -l 3s_h25
```
## Training

Usage:

```bash
python training.py -m model_name -o optimizer_name [-l] -c channel_name -t dataset_type -li window_configuration

```

The option `-l` states for loading data into RAM at the beginning of the experiment instead of reading it batch-by-batch from the disk. The dataset_types are : {original, harmonic and residual}. The channel_names are : {left, right, mid, side}. The window_configurations are : {window_size}_h{percentage_hop} for which datasets have already been extracted.

For example, 
```bash
python training.py -m han16 -l -c left -t residual -li 3s_h25
```

## Evaluation
 
Usage:
 
```bash
python evaluation.py -m model_name -w /path/to/weights/file.hdf5 -s evaluation_strategy -c channel_name -t dataset_type -l window_configuration
```

The weights for the reported models can be found at `./weights/model_name` folder. The dataset_types are : {original, harmonic and residual}. The channel_names are : {left, right, mid, side}. The window_configurations are : {window_size}_h{percentage_hop} for which datasets have already been extracted.

Evaluation strategies are:
The `s1` strategy computes a mean activation through whole audio excerpt and apply identification threshold

For example,
```bash
python evaluation.py -m han16 -w /homedtic/vshenoykadandale/DeepLearning/weights/3s_h25/residual/left/epoch.20-val_loss.0.334-fbeta.0.751-han_base_model.hdf5 -s s1 -c left -t residual -l 3s_h25
```

## References
* han16 model published by Han, Y., Kim, J., Lee, K., Han, Y., Kim, J., & Lee, K. (2017). Deep convolutional neural networks for predominant instrument recognition in polyphonic music. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 25(1), 208-221.  
* Source Code largely adapted from https://github.com/Veleslavia/EUSIPCO2017/ which provides implementation of han16 model along with data preprocessing, training and evaluation for IRMAS dataset.  
* The above mentioned source code was published in Pons, J., Slizovskaia, O., Gong, R., Gómez, E., & Serra, X. (2017, August). Timbre analysis of music audio signals with convolutional neural networks. In Signal Processing Conference (EUSIPCO), 2017 25th European (pp. 2744-2748). IEEE.  
