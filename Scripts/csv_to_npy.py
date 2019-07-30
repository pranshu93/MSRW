import numpy as np
import pandas as pd
import sys
import os

data_dirs = ['/scratch/sk7898/austere/classification_data_windowed/winlen_256_winindex_all/pedbike_class_winlen_256_winindex_all',
             '/scratch/sk7898/austere/classification_data_windowed/winlen_384_winindex_all/pedbike_class_winlen_384_winindex_all',
             '/scratch/sk7898/austere/classification_data_windowed/winlen_512_winindex_all/pedbike_class_winlen_512_winindex_all']

for data_dir in data_dirs:
    train_csv = os.path.join(data_dir, os.path.basename(data_dir)+'_train_freq.csv')
    test_csv = os.path.join(data_dir, os.path.basename(data_dir)+'_test_freq.csv')
    val_csv = os.path.join(data_dir, os.path.basename(data_dir)+'_val_freq.csv')

    train = pd.read_csv(train_csv); 
    test = pd.read_csv(test_csv);
    val = pd.read_csv(val_csv);

    X_train = (train.iloc[:,:train.shape[1]-1].values).astype('float32')
    Y_train = train.iloc[:,-1].values.astype('int32')

    X_test = (test.iloc[:,:test.shape[1]-1].values).astype('float32')
    Y_test = test.iloc[:,-1].values.astype('int32')

    X_val = (val.iloc[:,:val.shape[1]-1].values).astype('float32')
    Y_val = val.iloc[:,-1].values.astype('int32')

    # Save train data
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_train.npy"), X_train)
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_train_lbls.npy"), Y_train)

    # Save test data
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_test.npy"), X_test)
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_test_lbls.npy"), Y_test)

    # Save validation data
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_val.npy"), X_val)
    np.save(os.path.join(data_dir, os.path.basename(data_dir) + "_freq_val_lbls.npy"), Y_val)


