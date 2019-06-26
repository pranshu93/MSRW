import numpy as np
import pandas as pd
import sys
import math
from sklearn.metrics import recall_score
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten
from keras.constraints import maxnorm
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from keras.initializers import glorot_normal, uniform, normal
import tensorflow as tf
tf.set_random_seed(42)

# Do not allocate all the memory for visible GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config, graph=tf.get_default_graph())
K.set_session(sess)


test_run = False
NUM_CLASSES = 3
data_dir_prefix = '/scratch/dr2915/Bumblebee/bb_3class_winlen_256_winindex_all/'
#train_dir_prefix = 'bb_3class_winlen_256_winindex_all_train'
#test_dir_prefix = 'bb_3class_winlen_256_winindex_all_test'
#val_dir_prefix = 'bb_3class_winlen_256_winindex_all_val'
#classes = ['Noise', 'Human', 'Nonhuman']

test = pd.read_csv(os.path.join(data_dir_prefix, '3class_winlen_256_test.csv'), header=None);

colNames = ['feat_'+str(x) for x in range(0, test.shape[1])]

train = pd.read_csv(os.path.join(data_dir_prefix, '3class_winlen_256_train.csv'), names=colNames);

val = pd.read_csv(os.path.join(data_dir_prefix, '3class_winlen_256_val.csv'), names=colNames);

train = train.sample(frac=1, random_state=42)
train_val = train.append(val, ignore_index=True) #pd.concat([train, val], axis = 0, ignore_index=True)

X_train_val = (train_val.iloc[:,:train_val.shape[1]-1].values).astype('float32')
Y_train_val = train_val.iloc[:,-1].values.astype('int32')

X_test = (test.iloc[:,:test.shape[1]-1].values).astype('float32')
Y_test = test.iloc[:,-1].values.astype('int32')

#Y_train_val = Y_train_val.astype('float')
#Y_test = Y_test.astype('float')

#X_val = (val.iloc[:,:val.shape[1]-1].values).astype('float32')
#Y_val = val.iloc[:,-1].values.astype('int32')
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print('X_train_val shape: ', X_train_val.shape)
print('Y_train_val shape: ', Y_train_val.shape)
#print('Train_Val shape: ', train_val.shape)

def build_model(filters_1 = 64, filters_2 = 64, pool_size_1 = (1, 4), pool_size_2 = (1, 4), \
                fc_units_1 = 32, fc_units_2 = 32, dense_activation = 'relu', dropout=0.2, optimizer = 'Adam'):

    if optimizer == 'Adam':
        _optimizer = Adam(lr=0.0001)
    elif optimizer == 'SGD':
        _optimizer = SGD(lr=0.001)
    
    model = Sequential()
    model.add(Conv2D(filters_1, kernel_size=pool_size_1, strides=(1, 2), input_shape=(windows, window_dim, 1),\
                     padding='same', activation='relu', kernel_constraint=maxnorm(3), data_format='channels_last'))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters_2, kernel_size=pool_size_2, strides=(1, 2), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(1, 4)))
    model.add(Flatten())
    model.add(Dense(fc_units_1, activation=dense_activation, kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(fc_units_2, activation=dense_activation, kernel_constraint=maxnorm(3)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=_optimizer, metrics=['accuracy'])
    return model

def scheduler(epoch):
    initial_lrate = 0.001
    if epoch%5 == 0:
        lrate = initial_lrate * 0.01
    return lrate

windows = 32
window_dim = 16

X_train_val = X_train_val.reshape(X_train_val.shape[0], windows, window_dim, 1)
X_test = X_test.reshape(X_test.shape[0], windows, window_dim, 1)
#X_val = X_val.reshape(X_val.shape[0], windows, window_dim, 1)

lrate = LearningRateScheduler(scheduler)

model = KerasClassifier(build_fn = build_model, verbose=1)
if test_run:
    batch_size = [32]
    epochs = [1] 
    dropout = [0.2]
    filters1 = [16]
    filters2 = [16]
    pool_size_1 = [(1, 4)]
    pool_size_2 = [(1, 4)]
    nodes1 = [32]
    nodes2 = [10]
    dense_activation = ['relu']
    optimizer = ['SGD']
else:
    batch_size = [32, 64, 128]
    epochs = [10, 20, 40]
    dropout = [0.2, 0.3, 0.4]
    filters1 = [16, 32, 64]
    filters2 = [16, 32, 64]
    pool_size_1 = [(1, 4), (1, 8)]
    pool_size_2 = [(1, 4), (1, 8)]
    nodes1 = [32, 64, 128, 512]
    nodes2 = [10, 32, 64]
    dense_activation = ['relu', 'sigmoid', 'tanh']
    optimizer = ['SGD', 'Adam']

param_grid = dict(batch_size=batch_size, epochs=epochs, \
                  filters_1 = filters1, filters_2 = filters2, \
                  pool_size_1 = pool_size_1, pool_size_2 = pool_size_2, \
                  fc_units_1=nodes1, fc_units_2=nodes2, dense_activation = dense_activation,\
                  dropout = dropout, optimizer = optimizer)

validation_set_indices = [-1]*len(train) + [0]*len(val)
ps = PredefinedSplit(test_fold = validation_set_indices)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv = ps)
grid_result = grid.fit(X_train_val, Y_train_val)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

Y_pred = grid_result.predict(X_test)
print(Y_pred)
#Y_pred = np.argmax(y_prob, axis=1)
acc = grid_result.score(X_test, Y_test)
recall = recall_score(Y_test, Y_pred, average=None)

print('Test Accuracy:', acc)
print('Test Recall:', recall)
