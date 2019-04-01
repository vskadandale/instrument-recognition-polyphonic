import argparse
import os
import numpy as np
import pickle
import pandas as pd
import scipy.io as sio
from scipy.io import wavfile

import sys
sys.path.append('..')
from settings import *

"""
The wavfiles for each of the samples which were split using gen_split.py are generated using this code.
We generate them separately for each of the channels - left, right, mid and side.
We need it for feature extraction since Essentia's Music Extractor takes only audio file as input and not the sample array.
"""

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def create_wavfiles(data,path):
    create_folder(path)
    for i,e in enumerate(data):
        filename=os.path.join(path,str(i)+'.wav')
        e_float=np.float32(e)
        sio.wavfile.write(filename,44100,e_float.T)

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-c',
			 action='store',
                         dest='config',
                         help='-c type of the dataset. For ex: _1s_h100 for 1s with full length hop')
    args = aparser.parse_args()
    if not args.config:
        aparser.error('Please specify the data config!')
                
    
    source_path=PATH_TO_DATASET_SPLIT_FOLDERS+args.config
    dest_path=PATH_TO_ORIGINAL_WAV_FILES+args.config

    datasets=os.listdir(source_path)
    for dataset in datasets:
        folder_path=os.path.join(source_path,dataset,'X')
        files=os.listdir(folder_path)
        for filename in files:
            file_path=os.path.join(folder_path,filename)
            df=pd.read_csv(file_path,header=None)
            left_right_df=np.hsplit(df.values,2)
            left=left_right_df[0] #L
            right=left_right_df[1] #R
            mid=(left+right)/float(2) #M
            side=(left-right) #S
        
            left_path=os.path.join(dest_path,dataset,'left',filename.replace('.csv',''))
            right_path=os.path.join(dest_path,dataset,'right',filename.replace('.csv',''))
            mid_path=os.path.join(dest_path,dataset,'mid',filename.replace('.csv',''))
            side_path=os.path.join(dest_path,dataset,'side',filename.replace('.csv',''))

            create_wavfiles(left,left_path)
            create_wavfiles(right,right_path)
            create_wavfiles(mid,mid_path)
            create_wavfiles(side,side_path)

if __name__ == "__main__":
    main()

