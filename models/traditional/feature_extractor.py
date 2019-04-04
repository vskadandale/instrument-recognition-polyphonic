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
        del features_frames
    except RuntimeError as e:
        pass
        #print(path_to_wav_file + " is almost silent")
        #empty_files += 1
    path_to_features= path_to_wav_file.replace(os.path.basename(PATH_TO_WAV_FILES),'features').replace('.wav','.json')
    create_folder(os.path.dirname(path_to_features))
    es.YamlOutput(filename=path_to_features, format='json')(features)
    del features,path_to_features
    return 1

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-c',
                         action='store',
                         dest='channel',
                         help='-c channel in LRMS. Choose left/right/mid/side.')
    aparser.add_argument('-t',
                         action='store',
                         dest='type',
                         help='-t type of data. Choose between harmonic/original/residual.')

    args = aparser.parse_args()
    path_to_wavfiles = list(Path(os.path.join(PATH_TO_WAV_FILES,args.type,args.channel)).rglob("*.wav"))  ## Finds all the .wav_ files in the directory
    del args
    str_path_to_wavfiles=[i.as_posix() for i in path_to_wavfiles]
    del path_to_wavfiles
    pool=Pool(processes=23,maxtasksperchild=10)
    result=pool.map(extract_features,str_path_to_wavfiles,chunksize=512)
    del str_path_to_wavfiles,result
    pool.close() # no more tasks
    pool.join()
    #for path_to_wavfile in str_path_to_wavfiles:
    #    extract_features(path_to_wavfile)
    del pool

if __name__ == '__main__':
    main()
