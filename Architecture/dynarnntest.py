from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import os
import argparse
from math import sqrt
from tensorflow.contrib import rnn
from rnn import FastRNNCell,FastGRNNCell
import sys

# Making sure MSRW is part of python path
sys.path.insert(0, '../')
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

np.random.seed(42)
tf.set_random_seed(42)

class Bonsai:
    def __init__(self, x, C=2, F=16, P=5, D=2, S=1, lW=1, lT=1, lV=1, lZ=1,
                 sW=1, sT=1, sV=1, sZ=1, lr=None, W=None, T=None, V=None, Z=None):

        self.dataDimension = F + 1
        self.projectionDimension = P
        if (C > 2):
            self.numClasses = C
        elif (C == 2):
            self.numClasses = 1

        self.treeDepth = D
        self.sigma = S  # tf.Variable(S, name='sigma', dtype=tf.float32)
        self.lW = lW
        self.lV = lV
        self.lT = lT
        self.lZ = lZ
        self.sW = sW
        self.sV = sV
        self.sT = sT
        self.sZ = sZ

        self.internalNodes = 2 ** self.treeDepth - 1
        self.totalNodes = 2 * self.internalNodes + 1

        self.W = self.initW(W)
        self.V = self.initV(V)
        self.T = self.initT(T)
        self.Z = self.initZ(Z)

        self.W_th = tf.placeholder(tf.float32, name='W_th')
        self.V_th = tf.placeholder(tf.float32, name='V_th')
        self.Z_th = tf.placeholder(tf.float32, name='Z_th')
        self.T_th = tf.placeholder(tf.float32, name='T_th')

        self.W_st = tf.placeholder(tf.float32, name='W_st')
        self.V_st = tf.placeholder(tf.float32, name='V_st')
        self.Z_st = tf.placeholder(tf.float32, name='Z_st')
        self.T_st = tf.placeholder(tf.float32, name='T_st')

        if x is None:
            self.x = tf.placeholder("float", [None, self.dataDimension])
        else:
            self.x = x
        self.y = tf.placeholder("float", [None, self.numClasses])
        self.batch_th = tf.placeholder(tf.int64, name='batch_th')

        self.sigmaI = 1.0
        if lr is not None:
            self.learning_rate = lr
        else:
            self.learning_rate = 0.01

        self.hardThrsd()
        self.sparseTraining()
        self.lossGraph()
        self.trainGraph()
        self.accuracyGraph()

    def initZ(self, Z):
        if Z is None:
            Z = tf.random_normal([self.projectionDimension, self.dataDimension])
        Z = tf.Variable(Z, name='Z', dtype=tf.float32)
        return Z

    def initW(self, W):
        if W is None:
            W = tf.random_normal([self.numClasses * self.totalNodes, self.projectionDimension])
        W = tf.Variable(W, name='W', dtype=tf.float32)
        return W

    def initV(self, V):
        if V is None:
            V = tf.random_normal([self.numClasses * self.totalNodes, self.projectionDimension])
        V = tf.Variable(V, name='V', dtype=tf.float32)
        return V

    def initT(self, T):
        if T is None:
            T = tf.random_normal([self.internalNodes, self.projectionDimension])
        T = tf.Variable(T, name='T', dtype=tf.float32)
        return T

    # Commented by Dhrubo
    '''
    def getModelSize(self):
        nnzZ = np.ceil(int(Z.shape[0]*Z.shape[1])*sZ)
        nnzW = np.ceil(int(W.shape[0]*W.shape[1])*sW)
        nnzV = np.ceil(int(V.shape[0]*V.shape[1])*sV)
        nnzT = np.ceil(int(T.shape[0]*T.shape[1])*sT)
        return (nnzZ+nnzT+nnzV+nnzW)*8
    '''

    def getModelSize(self):
        nnzZ = np.ceil(int(self.Z.shape[0] * self.Z.shape[1]) * self.sZ)
        nnzW = np.ceil(int(self.W.shape[0] * self.W.shape[1]) * self.sW)
        nnzV = np.ceil(int(self.V.shape[0] * self.V.shape[1]) * self.sV)
        nnzT = np.ceil(int(self.T.shape[0] * self.T.shape[1]) * self.sT)
        return int((nnzZ + nnzT + nnzV + nnzW) * 8)


    def bonsaiGraph(self, X):
        X = tf.reshape(X, [-1, self.dataDimension])
        X_ = tf.divide(tf.matmul(self.Z, X, transpose_b=True), self.projectionDimension)
        # X_ = tf.nn.l2_normalize(tf.matmul(self.Z, X, transpose_b=True), 0)
        W_ = self.W[0:(self.numClasses)]
        V_ = self.V[0:(self.numClasses)]
        self.nodeProb = []
        self.nodeProb.append(1)
        score_ = self.nodeProb[0] * tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_)))
        for i in range(1, self.totalNodes):
            W_ = self.W[i * self.numClasses:((i + 1) * self.numClasses)]
            V_ = self.V[i * self.numClasses:((i + 1) * self.numClasses)]
            prob = (1 + ((-1) ** (i + 1)) * tf.tanh(tf.multiply(self.sigmaI,
                                                                tf.matmul(tf.reshape(self.T[int(np.ceil(i / 2) - 1)],
                                                                                     [-1, self.projectionDimension]),
                                                                          X_))))
            prob = tf.divide(prob, 2)
            prob = self.nodeProb[int(np.ceil(i / 2) - 1)] * prob
            self.nodeProb.append(prob)
            score_ = score_ + self.nodeProb[i] * tf.multiply(tf.matmul(W_, X_), tf.tanh(self.sigma * tf.matmul(V_, X_)))

        return score_, X_, self.T, self.W, self.V, self.Z

    def hardThrsd(self):
        self.W_op1 = self.W.assign(self.W_th)
        self.V_op1 = self.V.assign(self.V_th)
        self.T_op1 = self.T.assign(self.T_th)
        self.Z_op1 = self.Z.assign(self.Z_th)
        self.hard_thrsd_grp = tf.group(self.W_op1, self.V_op1, self.T_op1, self.Z_op1)

    def sparseTraining(self):
        self.W_op2 = self.W.assign(self.W_st)
        self.V_op2 = self.V.assign(self.V_st)
        self.Z_op2 = self.Z.assign(self.Z_st)
        self.T_op2 = self.T.assign(self.T_st)
        self.sparse_retrain_grp = tf.group(self.W_op2, self.V_op2, self.T_op2, self.Z_op2)

    def lossGraph(self):
        self.score, self.X_eval, self.T_eval, self.W_eval, self.V_eval, self.Z_eval = self.bonsaiGraph(self.x)

        if (self.numClasses > 2):
            self.margin_loss = utils.multiClassHingeLoss(tf.transpose(self.score), tf.argmax(self.y, 1), self.batch_th)
            self.reg_loss = 0.5 * (self.lZ * tf.square(tf.norm(self.Z)) + self.lW * tf.square(tf.norm(self.W)) +
                                   self.lV * tf.square(tf.norm(self.V)) + self.lT * tf.square(tf.norm(self.T)))
            self.loss = self.margin_loss + self.reg_loss
        else:
            self.margin_loss = tf.reduce_mean(tf.nn.relu(1.0 - (2 * self.y - 1) * tf.transpose(self.score)))
            self.reg_loss = 0.5 * (self.lZ * tf.square(tf.norm(self.Z)) + self.lW * tf.square(tf.norm(self.W)) +
                                   self.lV * tf.square(tf.norm(self.V)) + self.lT * tf.square(tf.norm(self.T)))
            self.loss = self.margin_loss + self.reg_loss

    def trainGraph(self):
        self.train_stepW = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.W]))
        self.train_stepV = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.V]))
        self.train_stepT = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.T]))
        self.train_stepZ = (tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.Z]))

    def accuracyGraph(self):
        if (self.numClasses > 2):
            correct_prediction = tf.equal(tf.argmax(tf.transpose(self.score), 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            y_ = self.y * 2 - 1
            correct_prediction = tf.multiply(tf.transpose(self.score), y_)
            correct_prediction = tf.nn.relu(correct_prediction)
            correct_prediction = tf.ceil(tf.tanh(correct_prediction))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def dynamicRNNFeaturizer(x):
    x = tf.unstack(x, 768, 1)
    #rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
    rnn_cell = FastGRNNCell(16)
    #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.9)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * 768 + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, 16]), index)
    return outputs

X = tf.placeholder("float", [None, 768, 10])
seqlen = tf.placeholder(tf.int32, [None])
bonsaiObj = Bonsai(dynamicRNNFeaturizer(X))
print('sai')