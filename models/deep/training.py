import argparse
import importlib
import os
import pickle

import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
#from keras.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from models.deep.dataloader import DataGenerator
import sys
sys.path.append('..')

import sys
sys.path.append('../..')
from settings import *

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def binarize(dataframe,threshold):
    df=dataframe.copy()
    mask1=df>threshold
    mask2=df==threshold
    mask3=df<threshold
    
    df.iloc[mask1]=1
    df.iloc[mask2]=1
    df.iloc[mask3]=0
    return df

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Trainer(object):
    init_lr = 0.001

    def __init__(self, X_train, X_val, y_train, y_val, model_module, optimizer, load_to_memory,channel,data_type,config):
        self.model_module = model_module
        #self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_module.BASE_NAME)))
        #self.dataset_mean = np.load('/homedtic/vshenoykadandale/DeepLearning/means/han16_medley_mean.npy')
        self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH,data_type,channel,'mean.npy'))
        self.channel=channel
        self.data_type=data_type
        self.config=config
        self.optimizer = optimizer if optimizer != 'sgd' else SGD(lr=self.init_lr, momentum=0.9, nesterov=True)
        self.in_memory_data = load_to_memory
        extended_x_train, extended_y_train = self._get_extended_data(X_train, y_train)
        extended_x_val, extended_y_val = self._get_extended_data(X_val, y_val)
        self.y_train = extended_y_train
        self.y_val = extended_y_val
        if self.in_memory_data:
            self.X_train = self._load_features(extended_x_train)
            self.X_val = self._load_features(extended_x_val)
        else:
            self.X_train = extended_x_train
            self.X_val = extended_x_val

    def _load_features(self, filenames):
        features = list()
        for filename in filenames:
            #feature_filename = os.path.join(MEDLEY_TRAIN_FEATURE_BASEPATH,self.data_type,self.channel,
            #                                "{}.npy".format(filename))
            feature_filename = filename.replace(PATH_TO_WAV_FILES, MEDLEY_TRAIN_FEATURE_BASEPATH)+'.npy'
            feature = np.load(feature_filename)
            feature -= self.dataset_mean
            features.append(feature)

        if K.image_dim_ordering() == 'th':
            features = np.array(features).reshape(-1, 1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR)
        else:
            features = np.array(features).reshape(-1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR, 1)
        return features

    def _get_extended_data(self, inputs, targets):
        extended_inputs = list()
        if self.model_module.BASE_NAME=='han16':
            num_segments=int(self.config.split('s')[0])
        else:
            num_segments=self.model_module.N_SEGMENTS_PER_TRAINING_FILE
        for i in range(0, num_segments):
            extended_inputs.extend(['_'.join(list(x)) for x in zip(inputs, [str(i)]*len(inputs))])
        extended_inputs = np.array(extended_inputs)
        extended_targets = np.tile(np.array(targets).reshape(-1),
                                   num_segments).reshape(-1, MEDLEY_N_CLASSES)
        return extended_inputs, extended_targets

    def _batch_generator(self, inputs, targets):
        assert len(inputs) == len(targets)
        while True:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - BATCH_SIZE + 1, BATCH_SIZE):
                excerpt = indices[start_idx:start_idx + BATCH_SIZE]
                if self.in_memory_data:
                    yield inputs[excerpt], targets[excerpt]
                else:
                    yield self._load_features(inputs[excerpt]), targets[excerpt]

    def train(self):
        model = self.model_module.build_model(MEDLEY_N_CLASSES)

        early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
        weights_path="{weights_basepath}/{model_path}/{data_type}/{channel}/".format(
                         weights_basepath=MODEL_WEIGHT_BASEPATH,
                         model_path=self.config,
                         data_type=self.data_type,
                         channel=self.channel)
        create_folder(weights_path)
        save_clb = ModelCheckpoint(
            weights_path +
            "epoch.{epoch:02d}-val_loss.{val_loss:.3f}-fbeta.{val_fbeta_score:.3f}"+"-{key}.hdf5".format(
                key=self.model_module.MODEL_KEY),
            monitor='val_loss',
            save_best_only=True)
        lrs = LearningRateScheduler(lambda epoch_n: self.init_lr / (2**(epoch_n//SGD_LR_REDUCE)))
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', f1])

        history = model.fit_generator(#self._batch_generator(self.X_train, self.y_train),
                                      generator=DataGenerator(self.X_train,self.y_train,self.dataset_mean),
                                      samples_per_epoch=self.model_module.SAMPLES_PER_EPOCH,
                                      nb_epoch=MAX_EPOCH_NUM,
                                      verbose=2,
                                      use_multiprocessing=True,
                                      callbacks=[save_clb, early_stopping, lrs],
                                      #validation_data=self._batch_generator(self.X_val, self.y_val),
                                      validation_data=DataGenerator(self.X_val, self.y_val,self.dataset_mean),
                                      #nb_val_samples=self.model_module.SAMPLES_PER_VALIDATION,
                                      validation_steps=self.model_module.SAMPLES_PER_VALIDATION,
                                      class_weight=None,
                                      nb_worker=10)
        history_path='{history_basepath}/{model_path}/{data_type}/{channel}/'.format(
                         history_basepath=MODEL_HISTORY_BASEPATH,
                         model_path=self.config,
                         data_type=self.data_type,
                         channel=self.channel) 
        create_folder(history_path)
        pickle.dump(history.history, open(history_path+'history_{model_key}.pkl'.format(model_key=self.model_module.MODEL_KEY),'w'))


def main():
    #dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])
    #X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
    #                                                  to_categorical(np.array(dataset.class_id, dtype=int)),
    #                                                  test_size=VALIDATION_SPLIT)
    
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model to train')
    aparser.add_argument('-o',
                         action='store',
                         dest='optimizer',
                         default='sgd',
                         help='-o optimizer')
    aparser.add_argument('-l',
                         action='store_true',
                         dest='load_to_memory',
                         default=False,
                         help='-l load dataset to memory')
    aparser.add_argument('-c',
                         action='store',
                         dest='channel',
                         help='-c which channel : left/right/mid/side')
    aparser.add_argument('-t',
                         action='store',
                         dest='data_type',
                         help='-t type of data original/harmonic/residual')
    aparser.add_argument('-li',
                         action='store',
                         dest='config',
                         help='-li configuration of data. for ex: 3s_h25')
    args = aparser.parse_args()

    if not args.model:
        aparser.error('Please, specify the model to train!')
    try:
        if args.model in ALLOWED_MODELS:
            model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
            print("{} imported as 'model'".format(args.model))
        else:
            print("The specified model is not allowed")
    except ImportError as e:
        print(e)

    #filenames=np.load('/homedtic/vshenoykadandale/DeepLearning/metadata/medley_train_filenames.npy')
    filenames=np.load(os.path.join(PATH_TO_METADATA,args.data_type,args.channel,'medley_train_filenames.npy'))
    #labels=np.load('/homedtic/vshenoykadandale/DeepLearning/metadata/medley_train_labels_bin.npy')
    raw_labels=np.load(os.path.join(PATH_TO_METADATA,args.data_type,args.channel,'medley_train_labels.npy'))
    labels=(binarize(pd.DataFrame(raw_labels),0.4)).values
    X_train, X_val, y_train, y_val = train_test_split(list(filenames),labels,test_size=VALIDATION_SPLIT)
                        
    print("TRAIN VAL SPLIT SUCCESS!!!")

    trainer = Trainer(X_train, X_val, y_train, y_val, model_module, args.optimizer, args.load_to_memory,args.channel,args.data_type,args.config)
    trainer.train()


if __name__ == "__main__":
    main()
