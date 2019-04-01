import argparse
import importlib
import os
import numpy as np
import csv
import pandas as pd

import sys 
sys.path.append('../..')
from settings import *

"""
This code extracts the mel-spectrograms from the audio belonging to training set and stores them. 
These mel-spectrograms are the input to the CNNs.
A large part of this code is taken from https://github.com/Veleslavia/EUSIPCO2017/blob/master/preprocessing.py

"""

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder


def preprocess(model_module,config,data_type,channel):
    path_to_dataset=os.path.join(PATH_TO_WAV_FILES,data_type,'wavs_'+config)
    path_to_features=os.path.join(MEDLEY_TRAIN_FEATURE_BASEPATH,config,data_type,channel)
    create_folder(path_to_features)
    path_to_labels=PATH_TO_LABELS+config
    path_to_metadata=os.path.join(PATH_TO_METADATA,config,data_type,channel)
    create_folder(path_to_metadata)
    path_to_means_dir=os.path.join(MODEL_MEANS_BASEPATH,config,data_type,channel)
    create_folder(path_to_means_dir)
    path_to_means=os.path.join(path_to_means_dir,'mean')
    
    datasets=['dataset_0','dataset_1','dataset_2','dataset_3']
    list_of_files=np.array([])
    start=True
    rownum=0
    create_folder(path_to_features)
    labels=pd.DataFrame()
    for dataset in datasets:
        empty_files=0
        print("[Dataset] : " + dataset)
        folder_path=os.path.join(path_to_dataset,dataset)
        channel_path=os.path.join(folder_path,channel)
        sub_folders=sorted(os.listdir(channel_path),key=lambda x: int(os.path.splitext(x)[0]))
        for sub_folder in sub_folders:
            sub_folder_path=os.path.join(channel_path,sub_folder)
            files=sorted(os.listdir(sub_folder_path),key=lambda x: int(os.path.splitext(x)[0]))
            for filename in files:
                f_name="{dataset}_{sub_folder}_{filename}".format(dataset=dataset,sub_folder=sub_folder,filename=filename)
                list_of_files=np.append(list_of_files,f_name)
                for i, spec_segment in enumerate(model_module.compute_spectrograms(os.path.join(sub_folder_path, filename))):
                    feature_filename = os.path.join(path_to_features,"{f_name}_{segment_idx}".format(f_name=f_name,segment_idx=i))
                    if(start):
                        rowsum=np.copy(spec_segment)
                        start=False
                    else:        
                        rowsum=rowsum+spec_segment
                    np.save(feature_filename, spec_segment)
                    rownum+=1

            #Load labels in sequence into npy for each dataset_i
            csv_path=os.path.join(path_to_labels,dataset,'y',sub_folder+'.csv')
            temp_df=pd.read_csv(csv_path,header=None)
            labels=pd.concat([labels,temp_df], ignore_index=True)
            del temp_df
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
    preprocess(model_module,args.config,args.data_type,args.channel)


if __name__ == "__main__":
    main()
