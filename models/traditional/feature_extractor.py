import argparse
import os
import json
import numpy as np

import essentia
import essentia.standard as es
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool

import sys
sys.path.append('../..')
from settings import *

"""
This code makes use of Essentia's music extractor to extract temporal, spectral and cepstral features from the wav files and persist them.

"""
def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def extract_features(path_to_wav_file):
    features = essentia.Pool()
    try:
        # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
        features, features_frames = es.MusicExtractor(lowlevelSilentFrames='drop',
                                                      lowlevelFrameSize=2048,
                                                      lowlevelHopSize=1024,
                                                      lowlevelStats=['mean', 'stdev'])(path_to_wav_file)
        features_frames = []
    except RuntimeError as e:
        print(path_to_wav_file + " is almost silent")
        #empty_files += 1
    path_to_features= path_to_wav_file.replace(os.path.basename(PATH_TO_WAV_FILES),'features').replace('.wav','.json')
    create_folder(os.path.dirname(path_to_features))
    es.YamlOutput(filename=path_to_features, format='json')(features)
    features.clear()
    return 1

def main():

    path_to_wavfiles = list(Path(PATH_TO_WAV_FILES).rglob("*.wav"))  ## Finds all the .wav_ files in the directory
    str_path_to_wavfiles=[i.as_posix() for i in path_to_wavfiles]
    pool=Pool(11)
    result=pool.map(extract_features,str_path_to_wavfiles,chunksize=1)
    pool.close() # no more tasks
    pool.join()

if __name__ == '__main__':
    main()
