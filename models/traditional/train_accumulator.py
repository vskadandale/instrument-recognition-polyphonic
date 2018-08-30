import argparse
import io,math
import os,sys
import json
import numpy as np
import pickle

import pandas as pd

#external .py files
import json_flattener #for flatenning jsons

sys.path.append('../..')
from settings import *

"""
This code aggregates datasets {0,1,2,3} and their respective labels into training set. The datasets contain nan entries since
the Essentia's music extractor treated some of the samples as silent clips because of insufficient energy.
We are retaining the nan entires in each of the datasets since we need to maintain the same output size across all the datasets.
During evaluation (in regressor.py), we take special care of them - we remove the nan entries before sending them sklearn modules,
get the regression results and then put the nan values back in place so that we have output of same size across harmonic, 
residual and original datasets across all the channels - {left, right, mid and side} 

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
    
    conf=args.config
    if args.data_type=='original':
        features_path=PATH_TO_ORIGINAL_FEATURES+conf
        train_path=PATH_TO_ORIGINAL_TRAINSET+conf
    elif args.data_type=='residual':
        features_path=PATH_TO_RESIDUAL_FEATURES+conf
        train_path=PATH_TO_RESIDUAL_TRAINSET+conf
    elif args.data_type=='harmonic':
        features_path=PATH_TO_HARMONIC_FEATURES+conf
        train_path=PATH_TO_HARMONIC_TRAINSET+conf
    label_path=PATH_TO_DATASET_SPLIT_FOLDERS+conf
    channels=['left','right','mid','side']
    datasets=['dataset_0','dataset_1','dataset_2','dataset_3']
    features=[]

    first_time=True

    for ind,dataset in enumerate(datasets):
        #Load samples
        for channel in channels:
            k=0
            train_label_df=pd.DataFrame()
            dataset_path=os.path.join(features_path,dataset,channel)
            folders=sorted(os.listdir(dataset_path),key=lambda x: int(os.path.splitext(x)[0]))
            for folder in folders:
                folder_path=os.path.join(dataset_path,folder)
                files=sorted(os.listdir(folder_path),key=lambda x: int(os.path.splitext(x)[0]))
                for filename in files:
                    filepath=os.path.join(folder_path,filename)
                    jsonFile = io.open(filepath,"r",encoding="utf-8")
                    jsonToPython = json.loads(jsonFile.read(), strict=False)
                    flatJson = json_flattener.flatten_json(jsonToPython)
                    
                    #Gather metadata on encountering first json
                    if first_time:
                        keys=flatJson.keys()
                        for e in keys:
                            key=e.encode("ascii")
                            if 'lowlevel' in key:
                                features.append(key)
                        dictValues={}
                        train_data_df=pd.DataFrame(dictValues, columns=sorted(features)).astype('float32')
                    if k==0:
                        dictValues={}
                        train_data_df=pd.DataFrame(dictValues, columns=sorted(features)).astype('float32')
                    for index in range(len(features)):
                        dictValues[features[index]]=flatJson.get(features[index])
                    train_data_df.loc[k]=(dictValues)
                    del flatJson
                    k+=1
                    first_time=False



            train_data_dump_path=os.path.join(train_path,channel)
            create_folder(train_data_dump_path)
            with open(os.path.join(train_data_dump_path,'train_data_{}.pkl'.format(ind)), 'wb') as fp:
                pickle.dump(train_data_df.astype('float32'), fp)

            del train_data_df
            print("[TRAIN DATA] Features of "+dataset+" and channel "+channel+" pickled!")
            
            #Load labels
            label_dir_path=os.path.join(label_path,dataset,'y')
            label_files=sorted(os.listdir(label_dir_path),key=lambda x: int(os.path.splitext(x)[0]))
            for csv in label_files:
                csv_path=os.path.join(label_dir_path,csv)
                temp_df=pd.read_csv(csv_path,header=None)
                train_label_df=pd.concat([train_label_df,temp_df], ignore_index=True) 
                del temp_df


            train_label_dump_path=os.path.join(train_path,channel)
            create_folder(train_label_dump_path)
            with open(os.path.join(train_label_dump_path,'train_label_{}.pkl'.format(ind)), 'wb') as fp:
                pickle.dump(train_label_df, fp)


if __name__ == '__main__':
    main()
