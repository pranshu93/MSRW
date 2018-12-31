from __future__ import print_function
import pickle
import numpy as np
import sys
import os
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

# Input directory
data_pref = sys.argv[1]
# Num classes
nb_classes = 2
np.random.seed(42)  # for reproducibility
n_specbins = 256
hidden_units = int(sys.argv[2])  # 256
nb_epochs = int(sys.argv[3])  # 100
dropout_rate = float(sys.argv[4])  # 0.2
learning_rate = float(sys.argv[5])  # 1e-4
# Batch size
batch_size = int(sys.argv[6])  # 64
# Optimizer
if sys.argv[7].lower() == 'Adam'.lower():
    optimizer = Adam(lr=learning_rate)
elif sys.argv[7].lower() == 'SGD'.lower():
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
# LSTM/GRU?
rnn=sys.argv[8]

# Stacked?
if len(sys.argv) > 9 and sys.argv[9].lower().__contains__('Stack'.lower()):  # Last argument can be stack/stacked/none
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

# Output filenames
out_pref=os.path.split(data_pref)[-1] # output prefix
if stacked:
    out_fname = os.path.join(out_pref + '_' + rnn + '_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.out')
else:
    out_fname = os.path.join(out_pref + '_' + rnn + '_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.out')

model_path = os.path.join(os.getcwd(),'model', rnn)
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = os.path.join(model_path, out_fname.replace('out', 'h5py'))
print('MODEL PATH:', model_file)

# Add diagnostic line
print('OUTPUT FILENAME: ', out_fname)

# Load data from file full prefix
train_all=np.loadtxt(os.path.join(data_pref + "_RNNspectrogram.csv"), delimiter=",")
# Get train data
X_train = train_all[:,0:-1]
y_train = train_all[:,-1]
# y_train = np_utils.to_categorical(y_train, nb_classes) # Doesn't work with StratifiedKFold

# Standardize data
mean=np.mean(X_train,0)
std=np.std(X_train,0)
std[std[:]<0.00001]=1
X_train=(X_train-mean)/std

# Number of RNN steps in data
n_steps = X_train.shape[1] // n_specbins

# Function to create model, required for KerasClassifier
def create_model():
    # Initialize model
    model = Sequential()
    model.add(Reshape((-1, n_specbins), input_shape=(X_train.shape[1],)))
    # Add RNN unit
    if rnn.lower().__contains__('GRU'.lower()):
        model.add(GRU(kernel_initializer="uniform", input_shape=(n_steps, n_specbins),
                      recurrent_initializer="uniform", units=hidden_units, return_sequences=return_sequences))
    elif rnn.lower().__contains__('LSTM'.lower()):
        model.add(LSTM(kernel_initializer="uniform", input_shape=(n_steps, n_specbins),
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
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.000001, verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
#model_ckpt = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=0) # only works with val_loss

model = KerasClassifier(build_fn = create_model, verbose=1, batch_size=batch_size, epochs=nb_epochs)
                                                        #, callbacks=[reduce_lr, early_stop, model_ckpt])
# Begin training
print("Train...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(model, X_train, y_train, cv=kfold, fit_params={'callbacks': [reduce_lr, early_stop]})
mean_cv_score=results.mean()
print(mean_cv_score)

## Save crossvalidation score with params
# Create result string
results_list = [sys.argv[8], hidden_units, nb_epochs, dropout_rate, learning_rate, batch_size, optimizer, stacked, mean_cv_score]
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