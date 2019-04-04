import numpy as np
import keras
import sys
sys.path.append('../..')
from settings import  *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, targets, mean, dim=(128,43),
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.n_channels=n_channels
        self.dataset_mean=mean
        self.batch_size = BATCH_SIZE
        self.labels = targets
        self.list_IDs = inputs
        self.n_classes = MEDLEY_N_CLASSES
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        targets =[self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, targets)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, targets):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim,self.n_channels))
        y = np.empty((self.batch_size, MEDLEY_N_CLASSES))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            feature_filename = ID.replace(PATH_TO_WAV_FILES, MEDLEY_TRAIN_FEATURE_BASEPATH)+'.npy'
            feature = np.load(feature_filename)
            feature -= self.dataset_mean
            # Store sample
            X[i,] = feature[...,None]
            y[i,] = targets[i]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y
