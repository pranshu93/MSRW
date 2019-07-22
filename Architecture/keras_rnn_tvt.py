from __future__ import print_function
import pickle
import numpy as np
import sys
import os
import argparse
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Reshape, Dropout #InputLayer
from keras.layers.recurrent import LSTM, GRU
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import recall_score

tf.set_random_seed(42)
#run_meta = tf.RunMetadata()
# Do not allocate all the memory for visible GPU
g=tf.get_default_graph()
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
parser.add_argument('-ct', type=int, default=0, help='LSTM(0)/GRU(1)')
parser.add_argument('-w', type=int, default=32, help='Window Length')
parser.add_argument('-lr', type=float, default=0.01, help='Learning Rate of Optimisation')
parser.add_argument('-bs', type=int, default=128, help='Batch Size of Optimisation')
parser.add_argument('-ep', type=int, default=500, help='Number of epochs')
parser.add_argument('-hs', type=int, default=64, help='Hidden Layer Size')
parser.add_argument('-dr', type=float, default=0.2, help='Dropout rate')
parser.add_argument('-ot', type=int, default=0, help='Adam(0)/Momentum(1)')
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
if args.ot == 0:
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
model_file = os.path.join(model_path, os.path.split(out_fname.replace('out', 'h5py'))[-1])
print('MODEL PATH:', model_file)

# Add diagnostic line
print('OUTPUT FILENAME: ', out_fname)

# Get train data
X_train = np.load(os.path.join(base_dir, data_pref + "_train.npy"))
y_train = np.load(os.path.join(base_dir, data_pref + "_train_lbls.npy"))

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# Standardize data
mean=np.mean(X_train,0)
std=np.std(X_train,0)
std[std[:]<0.00001]=1
X_train=(X_train-mean)/std

# Get val data
X_val = np.load(os.path.join(base_dir, data_pref + "_val.npy"))
y_val = np.load(os.path.join(base_dir, data_pref + "_val_lbls.npy"))

print('X_val shape:', X_val.shape)
print('y_val shape:', y_val.shape)

# Standardize data
std[std[:]<0.00001]=1
X_val=(X_val-mean)/std

# Get test data
X_test = np.load(os.path.join(base_dir, data_pref + "_test.npy"))
y_test = np.load(os.path.join(base_dir, data_pref + "_test_lbls.npy"))

print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Standardize data
std[std[:]<0.00001]=1
X_test=(X_test-mean)/std

# Number of RNN steps in data
n_steps = X_train.shape[1] // input_dim

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes=nb_classes)
y_val = to_categorical(y_val, num_classes=nb_classes)
y_test = to_categorical(y_test, num_classes=nb_classes)

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
        model.add(Dropout(rate=dropout_rate, seed=42))
    # Add stacked layer (optional)
    if stacked:  # TODO: Make stacked hidden unit different
        model.add(GRU(kernel_initializer="uniform", recurrent_initializer="uniform", units=hidden_units))
        # Add second dropout layer (optional)
        if dropout:  # TODO: Make stacked dropout rate different
            model.add(Dropout(rate=dropout_rate, seed=1337))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Get #FLOPS
    run_meta = tf.RunMetadata()
    with g.as_default():
        # Profile model
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    #model.summary()
    return model, flops

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0001, verbose=1)
early_stop = EarlyStoppingAfterNEpochs(monitor='val_loss', patience=10, verbose=1, start_epoch=50)
model_ckpt = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=0)

# Begin training
print("Train...")

# Create model
model, num_flops = create_model()
model.summary()
# Fit
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stop, model_ckpt])

score, acc = model.evaluate(X_test, y_test)

# Create result string
results_list = [rnn, input_dim, hidden_units, dropout_rate, learning_rate, batch_size, opt_name, stacked, num_flops.total_float_ops, acc]

# Append class recalls
y_pred = np.argmax(model.predict(X_test),axis=1)
recalls = recall_score(np.argmax(y_test, axis=1), y_pred, average=None)
for r in recalls:
    results_list.append(r)

#Print to output file
out_handle = open(out_fname, "a")
# Write a line of output
out_handle.write('\t'.join(map(str, results_list)) + '\n')
out_handle.close()