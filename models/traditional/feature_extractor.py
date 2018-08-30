import argparse
import os
import json
import numpy as np

import essentia
import essentia.standard as es
import pandas as pd

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

def main():
    
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-c',
			 action='store',
			 dest='config',
                         help='-c type of the dataset. For ex: _1s_h100 for 1s with full length hop')
    aparser.add_argument('-t',
                         action='store',
                         dest='data_type',
                         help='-t type of data original/harmonic/residual')

    args = aparser.parse_args()
    if not args.config:
        aparser.error('Please specify the data config!')
    
    conf = args.config
    if args.data_type=='original':
	path_to_dataset=PATH_TO_ORIGINAL_WAV_FILES+conf
	path_to_features=PATH_TO_ORIGINAL_FEATURES+conf       
    if args.data_type=='residual':
	path_to_dataset=PATH_TO_RESIDUAL_WAV_FILES+conf
	path_to_features=PATH_TO_RESIDUAL_FEATURES+conf       
    if args.data_type=='harmonic':
	path_to_dataset=PATH_TO_HARMONIC_WAV_FILES+conf
	path_to_features=PATH_TO_HARMONIC_FEATURES+conf       

    datasets=sorted(os.listdir(path_to_dataset))
    for dataset in datasets:  
        empty_files=0
        print("[Dataset] : " + dataset)
        folder_path=os.path.join(path_to_dataset,dataset)
        lrms=sorted(os.listdir(folder_path))
        for channel in lrms:
            channel_path=os.path.join(folder_path,channel)
            sub_folders=sorted(os.listdir(channel_path))
            for sub_folder in sub_folders:
                sub_folder_path=os.path.join(channel_path,sub_folder)
                files=sorted(os.listdir(sub_folder_path))
                for filename in files:
                    filepath=os.path.join(sub_folder_path,filename)
                    features=essentia.Pool()
                    try:
                        # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
                        features, features_frames = es.MusicExtractor(lowlevelSilentFrames='drop',
                                                                    lowlevelFrameSize=2048,
                                                                    lowlevelHopSize=1024,
                                                                    lowlevelStats=['mean', 'stdev'])(filepath)
                        features_frames=[]
                    except RuntimeError,e:
                        print(filepath+" is almost silent")
                        empty_files+=1
                    dump_path=os.path.join(path_to_features,dataset,channel,sub_folder)
                    create_folder(dump_path)
                    es.YamlOutput(filename = os.path.join(dump_path,filename.replace('.wav','.json')), format='json')(features)
                    features=[]
                    filename=[]
        print("Feature Extraction Completed Successfully for "+dataset)
        print("Total number of empty file in "+ dataset+" is "+str(empty_files))   


if __name__ == '__main__':
    main()
