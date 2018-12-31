from __future__ import print_function
import pickle
import numpy as np
import sys
import os
import argparse
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Reshape, Dropout #InputLayer
from keras.layers.recurrent import LSTM, GRU
import tensorflow as tf
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

tf.set_random_seed(42)
# Do not allocate all the memory for visible GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

class EarlyStoppingAfterNEpochs(EarlyStopping):
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 50): # add argument for starting epoch
        super(EarlyStoppingAfterNEpochs, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

# Args
parser = argparse.ArgumentParser(description='HyperParameters for Keras RNN Algorithm')
parser.add_argument('-ct', type=int, default=1, help='LSTM(0)/GRU(1)')
parser.add_argument('-w', type=int, default=32, help='Window Length')
parser.add_argument('-lr', type=float, default=0.01, help='Learning Rate of Optimisation')
parser.add_argument('-bs', type=int, default=128, help='Batch Size of Optimisation')
parser.add_argument('-ep', type=int, default=100, help='Number of epochs')
parser.add_argument('-hs', type=int, default=16, help='Hidden Layer Size')
parser.add_argument('-dr', type=float, default=0.2, help='Dropout rate')
parser.add_argument('-ot', type=int, default=1, help='Adam(0)/Momentum(1)')
parser.add_argument('-st', type=int, default=0, help='Stacked? No(0)/Yes(1)')
parser.add_argument('-out', type=str, default='default.out', help='Output filename')
parser.add_argument('-pref', type=str, default='h=468_b=558_winlen=384_str=128', help='Data prefix')
parser.add_argument('-base', type=str, default='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/'
                                               'Research/Deep_Learning_Radar/TimeFreqRNN/Data/Austere/Activity/All/',
                    help='Base location of data')
parser.add_argument('-model', type=str, default='.', help='Model path')

args=parser.parse_args()

# Input directory
base_dir = args.base
# Data prefix
data_pref = args.pref
# Num classes
nb_classes = 2
np.random.seed(42)  # for reproducibility
input_dim = args.w
hidden_units = args.hs  # 256
nb_epochs = args.ep  # 100
dropout_rate = args.dr  # 0.2
learning_rate = args.lr  # 1e-4
# Batch size
batch_size = args.bs  # 64
# Optimizer
if args.ot == 1:
    optimizer = Adam(lr=learning_rate)
    opt_name='Adam'
else:
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    opt_name = 'SGD'
# LSTM/GRU?
if args.ct == 1:
    rnn='GRU'
else:
    rnn='LSTM'

# Stacked?
if args.st==1:  # Last argument can be stack/stacked/none
    stacked = True
    return_sequences = True
else:
    stacked = False
    return_sequences = False

# Dropout?
if dropout_rate == 0.0:
    dropout = True
else:
    dropout = False

# Output filename
out_fname=args.out

model_path = os.path.join(args.model,'model', rnn)
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = os.path.join(model_path, out_fname.replace('out', 'h5py'))
print('MODEL PATH:', model_file)

# Add diagnostic line
print('OUTPUT FILENAME: ', out_fname)

# Load data from file full prefix
train_all=np.loadtxt(os.path.join(base_dir, data_pref + "_RNNspectrogram.csv"), delimiter=",")
# Get train data
X_train = train_all[:,0:-1]
y_train = train_all[:,-1]

# Standardize data
mean=np.mean(X_train,0)
std=np.std(X_train,0)
std[std[:]<0.00001]=1
X_train=(X_train-mean)/std

# Number of RNN steps in data
n_steps = X_train.shape[1] // input_dim

# Function to create model, required for KerasClassifier
def create_model():
    # Initialize model
    model = Sequential()
    model.add(Reshape((-1, input_dim), input_shape=(X_train.shape[1],)))
    # Add RNN unit
    if rnn.lower().__contains__('GRU'.lower()):
        model.add(GRU(kernel_initializer="uniform", input_shape=(n_steps, input_dim),
                      recurrent_initializer="uniform", units=hidden_units, return_sequences=return_sequences))
    elif rnn.lower().__contains__('LSTM'.lower()):
        model.add(LSTM(kernel_initializer="uniform", input_shape=(n_steps, input_dim),
                       recurrent_initializer="uniform", units=hidden_units, unit_forget_bias=True))
    # Add dropout layer (optional)
    if dropout:
        model.add(Dropout(rate=dropout_rate, seed=1337))
    # Add stacked layer (optional)
    if stacked:  # TODO: Make stacked hidden unit different
        model.add(GRU(kernel_initializer="uniform", recurrent_initializer="uniform", units=hidden_units))
        # Add second dropout layer (optional)
        if dropout:  # TODO: Make stacked dropout rate different
            model.add(Dropout(rate=dropout_rate, seed=1337))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.000001, verbose=1)
early_stop = EarlyStoppingAfterNEpochs(monitor='val_loss', patience=10, verbose=1, start_epoch=50)
model_ckpt = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=0)

# Begin training
print("Train...")
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# results = cross_val_score(model, X_train, y_train, cv=kfold, fit_params={'callbacks': [reduce_lr, early_stop]})
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
results=np.array([0]*5)
for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)
    X_train_cv = X_train[train_idx]
    y_train_cv = np_utils.to_categorical(y_train[train_idx], nb_classes)
    X_valid_cv = X_train[val_idx]
    y_valid_cv = np_utils.to_categorical(y_train[val_idx], nb_classes)

    # Create model
    model = create_model()

    # Fit
    history = model.fit(X_train_cv, y_train_cv, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_valid_cv, y_valid_cv),
                        callbacks=[reduce_lr, early_stop, model_ckpt])

# Print CV accuracy
mean_cv_score = np.array(history.history['val_acc']).mean()
print('Mean 5-fold CV score=', mean_cv_score)

## Save crossvalidation score with params
# Create result string
results_list = [sys.argv[8], hidden_units, nb_epochs, dropout_rate, learning_rate, batch_size, opt_name, stacked, mean_cv_score]
# Print to output file
out_handle = open(out_fname, "a")
# Write a line of output
out_handle.write('\t'.join(map(str, results_list)) + '\n')
out_handle.close()

'''
# Fit
#history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs,
#                    callbacks=[reduce_lr, early_stop, model_ckpt])
# Append params to history
# history.params["lr"] = learning_rate
# history.params["dr"] = dropout_rate
# history.params["n_hid"] = hidden_units
# history.params["opt"] = sys.argv[7]

# Begin testing
#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)
# Append test details to history
#history.history["test_loss"] = score
#history.history["test_acc"] = acc

# Save history and params
#with open(history_fname, 'wb') as file_pi:
#    pickle.dump(history.history, file_pi)
'''