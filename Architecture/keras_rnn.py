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

# Input directory
in_path = sys.argv[1]
# Num classes
nb_classes = 2
np.random.seed(42)  # for reproducibility

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
# Stacked?
if len(sys.argv) > 10 and sys.argv[10].lower().__contains__('Stack'.lower()):  # Last argument can be stack/stacked/none
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
if stacked:
    history_fname = 'stackedhist_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.pkl'
else:
    history_fname = 'hist_h=' + sys.argv[2] + '_e=' + sys.argv[3] + '_d=' + sys.argv[4] + '_l=' + sys.argv[
        5] + '_b=' + sys.argv[6] + '_' + sys.argv[7].lower() + '.pkl'

model_path = os.path.join(os.getcwd(),'model', sys.argv[8])
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = os.path.join(model_path, history_fname.replace('pkl', 'h5py'))
print('MODEL PATH:', model_file)

# Add diagnostic line
print('OUTPUT FILENAME: ', history_fname)

# Load data from .npy data/label file full prefix
data_pref=sys.argv[9]


# Initialize model
model = Sequential()
model.add(Reshape((-1, n_mels), input_shape=(X_train.shape[1],)))

# Add RNN unit
if sys.argv[8].lower().__contains__('GRU'.lower()):
    model.add(GRU(kernel_initializer="uniform", input_shape=(n_steps, n_mels),
              recurrent_initializer="uniform", units=hidden_units, return_sequences=return_sequences))
elif sys.argv[8].lower().__contains__('LSTM'.lower()):
    model.add(LSTM(kernel_initializer="uniform", input_shape=(n_steps, n_mels),
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
model.summary()

# Begin training
print("Train...")

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.000001, verbose=1)
early_stop = EarlyStoppingAfterNEpochs(monitor='val_loss', patience=10, verbose=1, start_epoch=50)
model_ckpt = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=0)

# Fit
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stop, model_ckpt])
# Append params to history
# history.params["lr"] = learning_rate
# history.params["dr"] = dropout_rate
# history.params["n_hid"] = hidden_units
# history.params["opt"] = sys.argv[7]

# Begin testing
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
# Append test details to history
history.history["test_loss"] = score
history.history["test_acc"] = acc

# Save history and params
with open(history_fname, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)