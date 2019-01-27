from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import timeit
import os
import sys

tf.set_random_seed(42)
np.random.seed(42)

def explbl(labels):
    labels = np.array(labels - np.min(labels),dtype=int)
    nc = int(np.max(labels) - np.min(labels) + 1)
    cr_labels = np.zeros((labels.__len__(), nc))
    cr_labels[np.arange(labels.__len__()),np.array(labels.tolist(),dtype=int)] = 1
    return cr_labels

class FC:
    def __init__(self,hD,nc):
        self.hD = hD; self.nc = nc;
        self.FCW = tf.Variable(tf.random_normal([hD, nc],0.0,0.1)) 
        self.FCB = tf.Variable(tf.random_normal([nc],0.0,0.1)) 
        
    def compute(self, x):        
        return tf.matmul(x, self.FCW) + self.FCB

class RNN:
    def __init__(self,ts,iD,hD,nc):
        self.ts = ts; self.iD = iD; self.hD = hD; self.nc = nc;
        
        self.W = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.U = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.B = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
    def compute(self, x):
        state = tf.zeros([tf.shape(x)[0],self.hD])
        for i in range(self.ts):
            state=tf.tanh(tf.matmul(x[:,i,:], self.W)+tf.matmul(state,self.U)+self.B)
        return state

class GRU:
    def __init__(self,ts,iD,hD,nc):
        self.ts = ts; self.iD = iD; self.hD = hD; self.nc = nc;
        
        self.Wr = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Ur = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Br = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
        self.Wz = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Uz = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bz = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
        self.W = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.U = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.B = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
    def compute(self, x):
        state = tf.zeros([tf.shape(x)[0],self.hD])
        for i in range(self.ts):
            r = tf.sigmoid(tf.matmul(x[:,i,:], self.Wr) + tf.matmul(state, self.Ur) + self.Br)
            z = tf.sigmoid(tf.matmul(x[:,i,:], self.Wz) + tf.matmul(state, self.Uz) + self.Bz)
            h = tf.tanh(tf.matmul(x[:,i,:], self.W) + tf.matmul(tf.multiply(r,state), self.U) + self.B)            
            state = tf.multiply(state,z) + tf.multiply(h,1-z)
        return state

class LSTM:
    def __init__(self,ts,iD,hD,nc):
        self.ts = ts; self.iD = iD; self.hD = hD; self.nc = nc;
        
        self.Wi = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Ui = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bi = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
        self.Wf = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Uf = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bf = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
        self.Wo = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Uo = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bo = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
        self.Wg = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.Ug = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bg = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        
    def compute(self, x):
        state = tf.zeros([tf.shape(x)[0],self.hD])
        cell_state = tf.zeros([tf.shape(x)[0],self.hD])
        for i in range(self.ts):
            I = tf.sigmoid(tf.matmul(x[:,i,:], self.Wi) + tf.matmul(state, self.Ui) + self.Bi)
            f = tf.sigmoid(tf.matmul(x[:,i,:], self.Wf) + tf.matmul(state, self.Uf) + self.Bf)
            o = tf.sigmoid(tf.matmul(x[:,i,:], self.Wo) + tf.matmul(state, self.Uo) + self.Bo)
            g = tf.tanh(tf.matmul(x[:,i,:], self.Wg) + tf.matmul(state, self.Ug) + self.Bg)            
            cell_state = tf.multiply(cell_state,f) + tf.multiply(I,g)
            state = tf.multiply(tf.tanh(cell_state),o) 
        return state

class FASTRNN:
    def __init__(self,ts,iD,hD,nc):
        self.ts = ts; self.iD = iD; self.hD = hD; self.nc = nc;
        
        self.W = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.U = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.B = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 

        #self.alpha = tf.Variable(tf.constant(-3.0)) 
        #self.beta = tf.Variable(tf.constant(3.0)) 

        self.alpha = tf.Variable(tf.random_normal([1],0.0,0.1)) 
        self.beta = tf.Variable(tf.random_normal([1],0.0,0.1)) 

    def compute(self, x):
        state = tf.zeros([tf.shape(x)[0],self.hD])
        for i in range(self.ts):
            h_ = tf.sigmoid(tf.matmul(x[:,i,:], self.W) + tf.matmul(state, self.U) + self.B)
            state = tf.sigmoid(self.beta) * h_ + tf.sigmoid(self.alpha) * state
        return state

class FASTGRNN:
    def __init__(self,ts,iD,hD,nc):
        self.ts = ts; self.iD = iD; self.hD = hD; self.nc = nc;
        
        self.W = tf.Variable(tf.random_normal([self.iD, self.hD],0.0,0.1)) 
        self.U = tf.Variable(tf.random_normal([self.hD, self.hD],0.0,0.1)) 
        self.Bz = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 
        self.Bh = tf.Variable(tf.random_normal([self.hD],0.0,0.1)) 

        #self.zeta = tf.Variable(tf.constant(1.0)) 
        #self.nu = tf.Variable(tf.constant(-4.0)) 

        self.zeta = tf.Variable(tf.random_normal([1],0.0,0.1)) 
        self.nu = tf.Variable(tf.random_normal([1],0.0,0.1)) 

    def compute(self, x):
        state = tf.zeros([tf.shape(x)[0],self.hD])
        for i in range(self.ts):
            z = tf.sigmoid(tf.matmul(x[:,i,:], self.W) + tf.matmul(state, self.U) + self.Bz)
            h_ = tf.tanh(tf.matmul(x[:,i,:], self.W) + tf.matmul(state, self.U) + self.Bh)
            state = tf.multiply(tf.sigmoid(self.zeta) * (1 - z) + tf.sigmoid(self.nu),h_) + tf.multiply(z,state)
        return state

class Bonsai:
    def __init__(self,h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sig,hP):

        self.h = h; self.int_n = 2**self.h - 1; self.tot_n = 2**(self.h + 1) - 1
        self.pd = pd; self.nf = nf; self.nc = nc

        self.lT = lT; self.lW = lW; self.lV = lV; self.lZ = lZ;
        self.sT = sT; self.sW = sW; self.sV = sV; self.sZ = sZ;

        self.sig = sig
        self.sigI = 1
        self.hP = hP

        if(self.hP):
            self.Z = tf.Variable(tf.random_normal([self.nf, self.pd],0.0,0.1))

        if(self.int_n > 0): self.T = tf.Variable(tf.random_normal([self.int_n, self.pd],0.0,0.1))
        self.V = tf.Variable(tf.random_normal([self.tot_n, self.pd, self.nc],0.0,0.1)) 
        self.W = tf.Variable(tf.random_normal([self.tot_n, self.pd, self.nc],0.0,0.1)) 

    def compute(self, x):
        if(self.hP):
            pp = tf.matmul(x,self.Z)/self.pd
        else:
            pp = x
        I = tf.ones([tf.shape(x)[0],1]) 
        for i in range(1,self.tot_n):
            j = int(np.floor((i + 1) / 2) - 1)
            prob=0.5*tf.expand_dims(I[:,j],1)*(1+pow(-1,(i+1)-2*(j+1))*tf.tanh(self.sigI*tf.matmul(pp,tf.expand_dims(self.T[j],1))))
            I = tf.concat((I,prob), axis=1)

        state = tf.reduce_sum((tf.einsum('ij,kjl->kil',pp,self.W)*tf.tanh(self.sig*tf.einsum('ij,kjl->kil',pp,self.V))*tf.expand_dims(tf.transpose(I),2)),0)
        return state


