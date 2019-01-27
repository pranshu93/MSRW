from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import os
from math import sqrt
from tensorflow.contrib import rnn
from rnn import FastRNNCell,FastGRNNCell
from utils import *
import sys
from structure import *

#os.environ["CUDA_VISIBLE_DEVICES"]=""

args = getArgs()
print(args.ot)

def forward_iter(data, labels, data_seqlen, index, code):
    batchx = data[index];  batchy = labels[index]; batchz = data_seqlen[index]
    if(code): sess.run(train_op, feed_dict={X: batchx, Y:batchy, seqlen: batchz, learning_rate: lr})
    else: return(sess.run(accuracy, feed_dict={X: batchx, Y: batchy, seqlen: batchz, learning_rate: lr}))

def dynamicRNN(x):
    x = tf.unstack(x, seq_max_len, 1)
    #rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
    if(args.ct):
        #rnn_cell = FastGRNNCell(hidden_dim, gate_non_linearity=args.ggnl, update_non_linearity=args.gunl, wRank=int(args.wr*min(window,hidden_dim)), uRank=int(args.ur*hidden_dim))
        rnn_cell = FastGRNNCell(hidden_dim)
    else:
        rnn_cell = FastRNNCell(hidden_dim, update_non_linearity=args.unl, wRank=int(args.wr*min(window,hidden_dim)), uRank=int(args.ur*hidden_dim))
    #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.9)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.stack(outputs)	
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, hidden_dim]), index)
    return outputs
    #return tf.matmul(outputs, weights['out']) + biases['out']

def process(data,labels):
    cr_data = np.zeros((data.__len__(),seq_max_len,window)); cr_seqlen = [];
    cr_labels = np.zeros((data.__len__(), num_classes)); cr_labels[np.arange(data.__len__()),np.array(labels.tolist(),dtype=int)] = 1;
    for i in range(data.__len__()):
        num_iter = min(int(np.ceil(float(data[i].__len__()-window)/stride)),seq_max_len)
        st = 0 #int((int(np.ceil(float(data[i].__len__()-window)/stride))-num_iter) * 0.5)
        cr_seqlen.append(num_iter)
        for j in range(num_iter):cr_data[i][j] = data[i][slice(int((st+j)*stride),int((st+j)*stride+window))];
    cr_data = cr_data[np.array(cr_seqlen) > 1]; cr_labels = cr_labels[np.array(cr_seqlen) > 1];    
    return cr_data.tolist(), cr_labels.tolist(), cr_seqlen 

window = args.w
stride = int(window * args.sp); 

fileloc = os.path.abspath('../Datasets/Radar2/')
#tryloc = os.path.abspath('../Datasets/Austere/')
modelloc = os.path.abspath('../Models/')

#modelloc = "/scratch/cse/phd/anz178419/Models/MSRW/"

cv_ind = args.fn
train_cuts = []; train_cuts_lbls = [];
for i in range(5):
    if(i != cv_ind - 1):
        cuts = np.load(fileloc + "/f" + str(i) + "_cuts.npy"); train_cuts = np.concatenate([train_cuts,cuts]);
        labels = np.load(fileloc + "/f" + str(i) + "_cuts_lbls.npy"); train_cuts_lbls = np.concatenate([train_cuts_lbls,labels]);

test_cuts = np.load(fileloc + "/f" + str(cv_ind - 1) + "_cuts.npy"); test_cuts_lbls = np.load(fileloc + "/f" + str(cv_ind - 1) + "_cuts_lbls.npy")
#try_cuts = np.load(tryloc + "/f_cuts.npy"); try_cuts_lbls = np.load(tryloc + "/f_cuts_lbls.npy")

max_length = args.ml

#seq_max_len = int(np.floor(float(max_length-window)/stride)+1)
seq_max_len = int(np.ceil(float(max_length-window)/stride))

all_cuts = []; [all_cuts.extend(train_cuts[i]) for i in range(train_cuts.shape[0])];
mean = np.mean(np.array(all_cuts)); std = np.std(np.array(all_cuts));
#print(mean,std)
train_cuts_n = []; test_cuts_n = []; try_cuts_n = [];
[train_cuts_n.append(((np.array(train_cuts[i])-mean)/std).tolist()) for i in range(train_cuts.shape[0])]
[test_cuts_n.append(((np.array(test_cuts[i])-mean)/std).tolist()) for i in range(test_cuts.shape[0])]
#[try_cuts_n.append(((np.array(try_cuts[i])-mean)/std).tolist()) for i in range(try_cuts.shape[0])]

lr = args.lr

num_epochs = args.ne
batch_size = args.bs

hidden_dim = args.hs
num_classes = 2

h = args.dep
pd = args.P
nf = hidden_dim
nc = num_classes
lT = args.rT; lW = args.rW; lV = args.rV; lZ = args.rZ
sT = args.sT; sW = args.sW; sV = args.rV; sZ = args.sZ;
#lT = 1e-2; lW = 1e-2; lV = 1e-2; lZ = 1e-3
#sT = 0.5; sW = 0.5; sV = 0.5; sZ = 0.1;
sig = args.sig

train_data, train_labels, train_seqlen = process(train_cuts_n,train_cuts_lbls)
test_data, test_labels, test_seqlen = process(test_cuts_n,test_cuts_lbls)
#try_data, try_labels, try_seqlen = process(try_cuts_n,try_cuts_lbls)

tf.reset_default_graph()

X = tf.placeholder("float", [None, seq_max_len, window])
Y = tf.placeholder("float", [None, num_classes])

learning_rate = tf.placeholder("float", shape=(), name='learning_rate')
tf.set_random_seed(42)
np.random.seed(42)

seqlen = tf.placeholder(tf.int32, [None])

weights = { 'out': tf.Variable(tf.random_normal([hidden_dim, num_classes])) }
biases = { 'out': tf.Variable(tf.random_normal([num_classes])) }

if(args.dt):
    dc = Bonsai(h,pd,nf,nc,lT,lW,lV,lZ,sT,sW,sV,sZ,sig,args.hP)

def DC(x, clf_type = None):
    #dc = FC(hD,nc)
    outputs = dc.compute(x)
    return outputs

if(args.dt):
    logits = DC(dynamicRNN(X),None)
else:
    logits = tf.matmul(dynamicRNN(X), weights['out']) + biases['out']

prediction = tf.nn.softmax(logits)
pred_labels = tf.argmax(prediction, 1)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
if(args.ot):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True) 
else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(pred_labels, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables()))) 

saver = tf.train.Saver()

max_acc = 0; max_try_acc=0; best_iter = 0;

#if(args.dt):
#    dc.sigI = 1

for i in range(num_epochs):
    num_iter = int(train_data.__len__()/batch_size)
    [forward_iter(train_data,train_labels,train_seqlen,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(train_data,train_labels,train_seqlen,slice(num_iter*batch_size,train_data.__len__()),True)

    if(args.dt):
        sigI_old = dc.sigI; dc.sigI = 1e9;
        sigI_old *= 1.001
    acc = forward_iter(test_data,test_labels,test_seqlen,slice(0,test_data.__len__()),False)
    tr_acc = forward_iter(train_data,train_labels,train_seqlen,slice(0,train_data.__len__()),False)
    if(args.dt):
        dc.sigI = sigI_old
    #try_acc,_ = forward_iter(try_data,try_labels,try_seqlen,slice(0,try_data.__len__()),False)

    if(max_acc < acc):
        max_acc = acc
        saver.save(sess, modelloc + "/bestmodel.ckpt")	
    #if(max_try_acc < try_acc): max_try_acc = try_acc

        #saver.save(sess, modelloc + "bestmodel.ckpt")	

        #best_iter = i
    print(i,tr_acc,acc)
print(max_acc)
