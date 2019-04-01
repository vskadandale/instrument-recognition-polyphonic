import argparse
import librosa
import numpy as np
import os
import scipy.io as sio
from scipy.io import wavfile


import sys
sys.path.append('..')
from settings import *

"""
This code splits an audio file into harmonic component and residual component using librosa and stores them separately.
The split could be done in multiple configurations by specifying a margin parameter which we skip.
For more details, visit https://librosa.github.io/librosa/auto_examples/plot_hprss.html

"""

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-c',
			 action='store',
                         dest='config',
                         help='-c type of the dataset. For ex: _1s_h100 for 1s with full length hop')
    args = aparser.parse_args()
    if not args.config:
        aparser.error('Please specify the data config!')
                
    
    config=args.config
    original_path=PATH_TO_ORIGINAL_WAV_FILES+config
    harmonic_path=PATH_TO_HARMONIC_WAV_FILES+config
    create_folder(harmonic_path)
    residual_path=PATH_TO_RESIDUAL_WAV_FILES+config
    create_folder(residual_path)

    datasets=['dataset_0','dataset_1','dataset_2','dataset_3','dataset_4']
    channels=['left','right','mid','side']
    for dataset in datasets:
        for channel in channels:
            folder_path=os.path.join(original_path,dataset,channel)
            sub_folders=os.listdir(folder_path)
            for sub_folder in sub_folders:
                sub_folder_path=os.path.join(folder_path,sub_folder)
                wav_files=os.listdir(sub_folder_path)
                harmonic_wav_folder_path=os.path.join(harmonic_path,dataset,channel,sub_folder)
                create_folder(harmonic_wav_folder_path)
                residual_wav_folder_path=os.path.join(residual_path,dataset,channel,sub_folder)
                create_folder(residual_wav_folder_path)
                for wav_file in wav_files:
                    wav_file_path=os.path.join(sub_folder_path,wav_file)
                    y,sr=librosa.load(wav_file_path, sr=44100)
                    n = len(y)
                    n_fft = 2048
                    y_pad = librosa.util.fix_length(y, n + n_fft // 2)
                    D=librosa.stft(y_pad, n_fft=n_fft)
                    D_harmonic, D_percussive = librosa.decompose.hpss(D)
                    
                    y_harmonic=librosa.istft(D_harmonic, length=n)
                    y_residual=librosa.istft(D_percussive, length=n)
                    
                    harmonic_wav_file_path=os.path.join(harmonic_wav_folder_path,wav_file)
                    residual_wav_file_path=os.path.join(residual_wav_folder_path,wav_file)
                    
                    sio.wavfile.write(harmonic_wav_file_path,44100,y_harmonic)
                    sio.wavfile.write(residual_wav_file_path,44100,y_residual)
                    del y_harmonic,y_residual,y,D,D_harmonic,D_percussive,y_pad 

if __name__ == '__main__':
    main()
