import argparse
import io,math
import os,sys
import json
import numpy as np
import pickle

import pandas as pd

#external .py files
#import models.traditional.json_flattener as json_flattener #for flatenning jsons
import json_flattener#for flatenning jsons

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
                         dest='channel',
                         help='-c which channel : left/right/mid/side')
    aparser.add_argument('-t',
                         action='store',
                         dest='data_type',
                         help='-t type of data original/harmonic/residual')
    args = aparser.parse_args()
    if not args.channel:
        aparser.error('Please specify the data channel!')
    
    features_path = os.path.join(PATH_TO_FEATURES,args.data_type,args.channel)
    train_path = os.path.join(PATH_TO_TRAINSET,args.data_type,args.channel)

    datasets=['dataset_0','dataset_1','dataset_2','dataset_3']
    features=[]

    first_time=True

    for ind,dataset in enumerate(datasets):
        #Load samples
        k=0
        train_label_df=pd.DataFrame()
        print("[Dataset] : " + dataset)
        dataset_path = os.path.join(PATH_TO_LABELS, dataset + '.npy')
        data = np.load(dataset_path).item()

        for path in list(data['X']):
            partial_path = (path[0]).strip()
            filepath = os.path.join(features_path, partial_path[:-3]+'json')
            jsonFile = io.open(filepath,"r",encoding="utf-8")
            jsonToPython = json.loads(jsonFile.read(), strict=False)
            flatJson = json_flattener.flatten_json(jsonToPython)

            #Gather metadata on encountering first json
            if first_time:
                keys=flatJson.keys()
                for e in keys:
                    key=e.encode("ascii").decode("utf-8")
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

        train_label_df = pd.concat([train_label_df,pd.DataFrame(data['y'])],ignore_index=True)
        create_folder(train_path)
        with open(os.path.join(train_path,'train_data_{}.pkl'.format(ind)), 'wb') as fp:
            pickle.dump(train_data_df.astype('float32'), fp)

        del train_data_df
        print("[TRAIN DATA] Features of "+dataset+" and channel "+args.channel+" pickled!")

        with open(os.path.join(train_path,'train_label_{}.pkl'.format(ind)), 'wb') as fp:
            pickle.dump(train_label_df, fp)


if __name__ == '__main__':
    main()
