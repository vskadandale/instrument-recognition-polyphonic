import argparse
import os
import json
import itertools
import numpy as np
import pickle

import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.externals import joblib
import sys
sys.path.append('../..')
from settings import *

"""
This code fits a regressor on training dataset and then predicts the instrument annotations for the test set.
We make use of support vector regression (SVR).

"""
def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def aggregate_train_test_data(train_path,test_path,training_data,training_labels):
    dictValues={}
    training_data_df=pd.DataFrame(dictValues)
    training_labels_df=pd.DataFrame()
    for filename in training_data:
        with open (os.path.join(train_path,filename), 'rb') as fp:
            element_train_data_df = pickle.load(fp)
        training_data_df=pd.concat([training_data_df,element_train_data_df], ignore_index=True)
        del element_train_data_df
    
    for filename in training_labels:
        with open (os.path.join(train_path,filename), 'rb') as fp: 
            element_train_label_df = pickle.load(fp)
        training_labels_df=pd.concat([training_labels_df,element_train_label_df], ignore_index=True)
        del element_train_label_df

    with open (os.path.join(test_path,'test_data.pkl'), 'rb') as fp:
        test_data_df = pickle.load(fp)
    
    with open (os.path.join(test_path,'test_label.pkl'), 'rb') as fp:
        test_label_df = pickle.load(fp)
    
    return training_data_df,training_labels_df,test_data_df,test_label_df

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
    channels=['left','right','mid','side']
    training_data=['train_data_0.pkl','train_data_1.pkl','train_data_2.pkl','train_data_3.pkl']
    training_labels=['train_label_0.pkl','train_label_1.pkl','train_label_2.pkl','train_label_3.pkl']
    
    for channel in channels:
        
        if args.data_type=='original':
            train_path=os.path.join(PATH_TO_ORIGINAL_TRAINSET+conf,channel)
            test_path=os.path.join(PATH_TO_ORIGINAL_TESTSET+conf,channel)
            results_path=os.path.join(PATH_TO_ORIGINAL_RESULTS+conf,channel)
        if args.data_type=='residual':
            train_path=os.path.join(PATH_TO_RESIDUAL_TRAINSET+conf,channel)
            test_path=os.path.join(PATH_TO_RESIDUAL_TESTSET+conf,channel)
            results_path=os.path.join(PATH_TO_RESIDUAL_RESULTS+conf,channel)
        if args.data_type=='harmonic':
            train_path=os.path.join(PATH_TO_HARMONIC_TRAINSET+conf,channel)
            test_path=os.path.join(PATH_TO_HARMONIC_TESTSET+conf,channel)
            results_path=os.path.join(PATH_TO_HARMONIC_RESULTS+conf,channel)
        
        create_folder(results_path)

        x_train,y_train,x_test,y_test=aggregate_train_test_data(train_path,test_path,training_data,training_labels)
        
	#find row numbers where NaN entries are there ( in this case it turns out that either entire row is full of NaNs or no NaNs)
        nan_loc_train=np.where(x_train.isnull().any(axis=1)==True)[0]
        #drop NaN rows from samples
        filtered_train_data_df=x_train.dropna(axis=0, how='all')
        filtered_y_train=y_train.drop(y_train.index[nan_loc_train])

        scaler = StandardScaler() #To standardize the features to zero mean and unit variance
        x_train_scaled=scaler.fit_transform(filtered_train_data_df.values)
        
        svr = SVR()
        reg=MultiOutputRegressor(svr).fit(x_train_scaled, filtered_y_train)
        
        joblib.dump(reg, os.path.join(results_path,'model.pkl'))
        
        y_train_pred=reg.predict(x_train_scaled)
        
	#Insert NaN rows in predicted label matrix such that L R M S columns could be combined without any issues and then handle NaN
        y_train_pred_nan=y_train_pred.copy()
        nan_array = np.empty((1,MEDLEY_N_CLASSES,))
        nan_array.fill(np.nan)
        for pos in nan_loc_train:
            y_train_pred_nan=np.insert(y_train_pred_nan,pos,nan_array,0)
            y_train.iloc[pos]=nan_array

    
        #find row numbers where NaN entries are there ( in this case it turns out that either entire row is full of NaNs or no NaNs)
        nan_loc_test=np.where(x_test.isnull().any(axis=1)==True)[0]
        #drop NaN rows from samples
        filtered_test_data_df=x_test.dropna(axis=0, how='all')
         
        filtered_test_scaled=scaler.transform(filtered_test_data_df.values)
        y_test_pred=reg.predict(filtered_test_scaled)
        
        #Insert NaN rows in predicted label matrix such that L R M S columns could be combined without any issues and then handle NaN
        y_test_pred_nan=y_test_pred.copy()
        nan_array = np.empty((1,MEDLEY_N_CLASSES,))
        nan_array.fill(np.nan)
        for pos in nan_loc_test:
            y_test_pred_nan=np.insert(y_test_pred_nan,pos,nan_array,0)
            y_test.iloc[pos]=nan_array

        np.save(os.path.join(results_path,'train_truth'),y_train)
        np.save(os.path.join(results_path,'train_pred'),y_train_pred_nan)
        np.save(os.path.join(results_path,'test_truth'),y_test)
        np.save(os.path.join(results_path,'test_pred'),y_test_pred_nan)

if __name__ == '__main__':
    main()
