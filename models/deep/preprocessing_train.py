import argparse
import importlib
import os
import numpy as np
import csv
import pandas as pd

import sys 
sys.path.append('../..')
from settings import *
import  models.deep.experiments.models.han16
from models.deep.experiments.models.han16 import compute_spectrograms
"""
This code extracts the mel-spectrograms from the audio belonging to training set and stores them. 
These mel-spectrograms are the input to the CNNs.
A large part of this code is taken from https://github.com/Veleslavia/EUSIPCO2017/blob/master/preprocessing.py

"""

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder


def preprocess(model_module,data_type,channel):
    path_to_dataset=os.path.join(PATH_TO_WAV_FILES,data_type,channel)
    path_to_features=os.path.join(MEDLEY_TRAIN_FEATURE_BASEPATH,data_type,channel)
    create_folder(path_to_features)
    path_to_metadata=os.path.join(PATH_TO_METADATA,data_type,channel)
    create_folder(path_to_metadata)
    path_to_means_dir=os.path.join(MODEL_MEANS_BASEPATH,data_type,channel)
    create_folder(path_to_means_dir)
    path_to_means=os.path.join(path_to_means_dir,'mean')
    
    datasets=['dataset_0','dataset_1','dataset_2','dataset_3']
    list_of_files=np.array([])
    start=True
    rownum=0
    labels=pd.DataFrame()
    for dataset in datasets:
        print("[Dataset] : " + dataset)
        dataset_path=os.path.join(PATH_TO_LABELS,dataset+'.npy')
        data=np.load(dataset_path).item()

        for path in list(data['X']):
            partial_path=(path[0]).strip()
            filepath=os.path.join(path_to_dataset,partial_path)
            list_of_files=np.append(list_of_files,filepath)
            for i, spec_segment in enumerate(model_module.compute_spectrograms(filepath)):
                #feature_filename = os.path.join(path_to_features,partial_path[:-4])
                feature_filename = os.path.join(path_to_features,
                                                "{f_name}_{segment_idx}".format(f_name=partial_path, segment_idx=i))
                create_folder(os.path.dirname(feature_filename))
                if(start):
                    rowsum=np.copy(spec_segment)
                    start=False
                else:
                    rowsum=rowsum+spec_segment
                np.save(feature_filename, spec_segment)
                rownum+=1
        #Load labels in sequence into npy for each dataset_i
        labels=pd.concat([labels,pd.DataFrame(data['y'])], ignore_index=True)

    np.save(os.path.join(path_to_metadata,'medley_train_filenames'),list_of_files)
    np.save(os.path.join(path_to_metadata,'medley_train_labels'),labels)
    np.save(path_to_means,rowsum/float(rownum))

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model for preprocessing. At the moment, han16 is the only model that we have.')
    aparser.add_argument('-c',
                         action='store',
                         dest='channel',
                         help='-c which channel : left/right/mid/side')
    aparser.add_argument('-t',
                         action='store',
                         dest='data_type',
                         help='-t type of data original/harmonic/residual')
    aparser.add_argument('-l',
                         action='store',
                         dest='config',
                         help='-l configuration of data. for ex: 3s_h25')
    args = aparser.parse_args()

    if not args.model:
        aparser.error('Please, specify the model!')
    try:
        if args.model in ALLOWED_MODELS:
            model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
            print("{} imported as 'model'".format(args.model))
        else:
            print("The specified model is not allowed. At the moment, han16 is the only model that we have.")
    except ImportError as e:
        print(e)
    preprocess(model_module,args.data_type,args.channel)


if __name__ == "__main__":
    main()
