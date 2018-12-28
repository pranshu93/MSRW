
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Reshape, Flatten
from keras.constraints import maxnorm
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv('../Data/Bike_Human/bike_human.csv')
train = train.sample(frac=1)


# In[ ]:


X = (train.iloc[:,:train.shape[1]-1].values).astype('float32')
Y = train.iloc[:,-1].values.astype('int32')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 


# In[ ]:


def build_model(filters_1 = 64, filters_2 = 64, pool_size = (2, 2), fc_units_1 = 32, fc_units_2 = 32, dropout=0.2):
    model = Sequential()
    model.add(Conv2D(filters_1, kernel_size=(1, 8), strides=(1, 4), input_shape=(windows, window_dim, 1), padding='same', activation='relu', kernel_constraint=maxnorm(3), data_format='channels_last'))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters_2, kernel_size=(1, 4), strides=(1, 2), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(Reshape((5*6,32)))
    #model.add(LSTM(32, return_sequences=False))
    model.add(Flatten())
    model.add(Dense(fc_units_1, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(fc_units_2, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


def scheduler(epoch):
    initial_lrate = 0.001
    if epoch%5 == 0:
        lrate = initial_lrate * 0.01
    return lrate


# In[ ]:


windows = 10
window_dim = 50

X_train = X_train.reshape(X_train.shape[0], windows, window_dim, 1)
X_test = X_test.reshape(X_test.shape[0], windows, window_dim, 1)

lrate = LearningRateScheduler(scheduler)

model = KerasClassifier(build_fn = build_model, verbose=0)
batch_size = [10, 20]
epochs = [10]
dropout = [0.2, 0.3]
filters1 = [32, 64]
filters2 = [16, 32, 64]
pool_size = [(1, 4), (2, 2)]
nodes1 = [64, 128, 512]
nodes2 = [32, 10]
param_grid = dict(batch_size=batch_size, epochs=epochs, filters_1 = filters1, filters_2 = filters2, pool_size = pool_size, fc_units_1=nodes1, fc_units_2=nodes2, dropout = dropout)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv = 5) #sk_params={'callbacks': [lrate]}
grid_result = grid.fit(X_train, Y_train)


# In[ ]:


print "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print "%f (%f) with: %r" % (mean, stdev, param)


# In[ ]:


scores = model.evaluate(X_test, Y_test, verbose=0)
print 'Accuracy:', scores[1]*100


# In[ ]:




