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

#os.environ["CUDA_VISIBLE_DEVICES"]=""

def getArgs():
    parser = argparse.ArgumentParser(description='HyperParameters for Dynamic RNN Algorithm')
    parser.add_argument('-ct', type=int, default=1, help='FastRNN(False)/FastGRNN(True)')
    parser.add_argument('-unl', type=str, default="tanh" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
    parser.add_argument('-gunl', type=str, default="quantTanh" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
    parser.add_argument('-ggnl', type=str, default="quantSigm" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
    parser.add_argument('-ur', type=float, default=1, help='Rank of U matrix')
    parser.add_argument('-wr', type=float, default=1, help='Rank of W matrix')    
    parser.add_argument('-w', type=int, default=32, help='Window Length')
    parser.add_argument('-sp', type=float, default=0.5, help='Stride as % of Window Length(0.25/0.5/0.75/1)')
    parser.add_argument('-lr', type=float, default=0.01, help='Learning Rate of Optimisation')
    parser.add_argument('-bs', type=int, default=128, help='Batch Size of Optimisation')
    parser.add_argument('-hs', type=int, default=16, help='Hidden Layer Size')
    parser.add_argument('-ot', type=int, default=1, help='Adam(False)/Momentum(True)')
    parser.add_argument('-ml', type=int, default=768, help='Maximum slice length of cut taken for classification')
    parser.add_argument('-fn', type=int, default=3, help='Fold Number to classify for cross validation[1/2/3/4/5]')
    return parser.parse_args()

args = getArgs()
print(args.ot)

def forward_iter(data, labels, data_seqlen, index, code):
    batchx = data[index];  batchy = labels[index]; batchz = data_seqlen[index]
    if(code): sess.run(train_op, feed_dict={X: batchx, Y:batchy, seqlen: batchz, learning_rate: lr})
    else: return(sess.run([accuracy,reference], feed_dict={X: batchx, Y: batchy, seqlen: batchz, learning_rate: lr}))

def dynamicRNN(x):
    x = tf.unstack(x, seq_max_len, 1)
    #rnn_cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
    if(args.ct):
        rnn_cell = FastGRNNCell(hidden_dim, gate_non_linearity=args.ggnl, update_non_linearity=args.gunl, wRank=int(args.wr*min(window,hidden_dim)), uRank=int(args.ur*hidden_dim))
    else:
        rnn_cell = FastRNNCell(hidden_dim, update_non_linearity=args.unl, wRank=int(args.wr*min(window,hidden_dim)), uRank=int(args.ur*hidden_dim))
    #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,output_keep_prob=0.9)

    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.stack(outputs)	
    outputs = tf.transpose(outputs, [1, 0, 2])
    reference = outputs
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, hidden_dim]), index)
    return tf.matmul(outputs, weights['out']) + biases['out'], reference

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
tryloc = os.path.abspath('../Datasets/Austere/')
modelloc = os.path.abspath('../Models/')

cv_ind = args.fn
train_cuts = []; train_cuts_lbls = [];
for i in range(5):
    if(i != cv_ind - 1):
        cuts = np.load(fileloc + "/f" + str(i) + "_cuts.npy"); train_cuts = np.concatenate([train_cuts,cuts]);
        labels = np.load(fileloc + "/f" + str(i) + "_cuts_lbls.npy"); train_cuts_lbls = np.concatenate([train_cuts_lbls,labels]);

test_cuts = np.load(fileloc + "/f" + str(cv_ind - 1) + "_cuts.npy")
test_cuts_lbls = np.load(fileloc + "/f" + str(cv_ind - 1) + "_cuts_lbls.npy")
try_cuts = np.load(tryloc + "/f_cuts.npy")
try_cuts_lbls = np.load(tryloc + "/f_cuts_lbls.npy")


#max_length = 0;
#cut_lengths = []
#for i in range(train_cuts.shape[0]): cut_lengths.append(train_cuts[i].__len__());
#for i in range(test_cuts.shape[0]): cut_lengths.append(test_cuts[i].__len__());
#cut_lengths = np.array(cut_lengths)
max_length = args.ml#np.percentile(cut_lengths,0)
#print(max_length)

seq_max_len = int(np.floor(float(max_length-window)/stride)+1)

all_cuts = []; [all_cuts.extend(train_cuts[i]) for i in range(train_cuts.shape[0])];
mean = np.mean(np.array(all_cuts)); std = np.std(np.array(all_cuts));
print(mean,std)
train_cuts_n = []; test_cuts_n = []; try_cuts_n = [];
[train_cuts_n.append(((np.array(train_cuts[i])-mean)/std).tolist()) for i in range(train_cuts.shape[0])]
[test_cuts_n.append(((np.array(test_cuts[i])-mean)/std).tolist()) for i in range(test_cuts.shape[0])]
[try_cuts_n.append(((np.array(try_cuts[i])-mean)/std).tolist()) for i in range(try_cuts.shape[0])]

lr = args.lr

num_epochs = 500
batch_size = args.bs

hidden_dim = args.hs
num_classes = 2

train_data, train_labels, train_seqlen = process(train_cuts_n,train_cuts_lbls)
test_data, test_labels, test_seqlen = process(test_cuts_n,test_cuts_lbls)
try_data, try_labels, try_seqlen = process(try_cuts_n,try_cuts_lbls)

tf.reset_default_graph()

X = tf.placeholder("float", [None, seq_max_len, window])
Y = tf.placeholder("float", [None, num_classes])

learning_rate = tf.placeholder("float", shape=(), name='learning_rate')
tf.set_random_seed(42)
np.random.seed(42)

seqlen = tf.placeholder(tf.int32, [None])

weights = { 'out': tf.Variable(tf.random_normal([hidden_dim, num_classes])) }
biases = { 'out': tf.Variable(tf.random_normal([num_classes])) }

logits,reference = dynamicRNN(X)

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
'''
for i in range(num_epochs):
    num_iter = int(train_data.__len__()/batch_size)
    [forward_iter(train_data,train_labels,train_seqlen,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(train_data,train_labels,train_seqlen,slice(num_iter*batch_size,train_data.__len__()),True)
    acc = forward_iter(test_data,test_labels,test_seqlen,slice(0,test_data.__len__()),False)

    tr_acc = forward_iter(train_data,train_labels,train_seqlen,slice(0,train_data.__len__()),False)
    try_acc = forward_iter(try_data,try_labels,try_seqlen,slice(0,try_data.__len__()),False)

    if(max_acc < acc):
        max_acc = acc
        saver.save(sess, modelloc + "/bestmodel.ckpt")	

    if(max_try_acc < try_acc):	max_try_acc = try_acc

        #saver.save(sess, modelloc + "bestmodel.ckpt")	

        #best_iter = i
    #print(i,tr_acc,acc,try_acc)
print(max_acc,max_try_acc)
return [max_acc, max_try_acc]
'''
'''
saver.restore(sess, modelloc + "/bestmodel.ckpt")
#print(test_cuts[0][0:int(max_length)])
#print(test_cuts_n[0][0:int(max_length)])

acc,hdot = forward_iter(test_data,test_labels,test_seqlen,slice(0,1),False)
print(np.array(hdot,dtype='float'))

print()
print("The label for 1st test point is ",end="")
print(test_labels[0])

def formatp(v):
    if(v.ndim == 2):
        arrs = v.tolist()
        print("{",end="")
        for i in range(arrs.__len__()-1):
            print("{",end="")
            for j in range(arrs[i].__len__()-1):
                print("%.6f" % arrs[i][j],end=",")
            print("%.6f" % arrs[i][arrs[i].__len__()-1],end="},")
        print("{",end="")
        for j in range(arrs[arrs.__len__()-1].__len__()-1):
            print("%.6f" % arrs[arrs.__len__()-1][j],end=",")
        print("%.6f" % arrs[arrs.__len__()-1][arrs[arrs.__len__()-1].__len__()-1],end="}}")
    else:
        print(v)

#print(acc)
#print(formatp(np.array([test_cuts[1][0:int(max_length)]],dtype='float')))
print("Class probability values are ",end="")

print()

variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)
for k, v in zip(variables_names, values):
    np.save(modelloc + '/' + k.replace('/','_'),v)
    print ("Variable: ", k)
    #print ("Shape: ", v.shape)
    print (formatp(v))
    #print(np.min(v),np.max(v))
'''

qW1 = np.load(modelloc + "/QuantizedFastModel/qW1.npy") 
qFC_Bias = np.load(modelloc + "/QuantizedFastModel/qFC_Bias.npy") 
qW2 = np.load(modelloc + "/QuantizedFastModel/qW2.npy") 
qU2 = np.load(modelloc + "/QuantizedFastModel/qU2.npy") 
qFC_Weight = np.load(modelloc + "/QuantizedFastModel/qFC_Weight.npy") 
qU1 = np.load(modelloc + "/QuantizedFastModel/qU1.npy") 
qB_g = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_g.npy")) 
qB_h = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_h.npy"))
q = np.load(modelloc + "/QuantizedFastModel/paramScaleFactor.npy")

W1 = np.load(modelloc + "/W1.npy")
#np.array(qW1/q,dtype=float) #np.load(modelloc + "/W1.npy")
print(qW1/q)
print(W1)
FC_Bias = np.load(modelloc + "/FC_Bias.npy") 
W2 = np.load(modelloc + "/W2.npy") 
U2 = np.load(modelloc + "/U2.npy") 
FC_Weight = np.load(modelloc + "/FC_Weight.npy") 
U1 = np.load(modelloc + "/U1.npy") 
B_g = np.transpose(np.load(modelloc + "/B_g.npy")) 
B_h = np.transpose(np.load(modelloc + "/B_h.npy"))

zeta = np.load(modelloc + "/zeta.npy")
zeta = 1/(1 + np.exp(-zeta))
nu = np.load(modelloc + "/nu.npy")
nu = 1/(1 + np.exp(-nu))
I = 100000

def quantTanh(x,scale):
    return np.maximum(-scale,np.minimum(scale,x))

def quantSigm(x,scale):
    return np.maximum(np.minimum(0.5*(x+scale),scale),0)

def nonlin(code,x,scale):
    if(code == "t"): return quantTanh(x,scale)
    elif(code == "s"): return quantSigm(x,scale)

def predict_fp(points,lbls):
    pred_lbls = []
    
    for i in range(points.shape[0]):
        h = np.array(np.zeros((hidden_dim,1)),dtype=float)
        for t in range(seq_max_len):
            x=np.array(((np.array(points[i][slice(t*stride,t*stride+window)])-float(mean)))/float(std),dtype=float).reshape((-1,1))
            pre = np.array(np.matmul(np.transpose(np.matmul(W1,W2)),x) + np.matmul(np.transpose(np.matmul(U1,U2)),h),dtype=float)
            h_ = np.array(nonlin("t",pre+B_h,1),dtype=float)
            z = np.array(nonlin("s",pre+B_g,1),dtype=float)
            h = np.array((np.multiply(z,h) + np.array(np.multiply(zeta*(1-z)+nu,h_),dtype=float)),dtype=float)
        
        pred_lbls.append(np.argmax(np.matmul(np.transpose(h),FC_Weight) + FC_Bias))
    pred_lbls = np.array(pred_lbls)
    print(lbls)
    print(pred_lbls)
    print(float((pred_lbls==lbls).sum())/lbls.shape[0])

fpt = int

def predict(points,lbls):
    pred_lbls = []
    
    for i in range(points.shape[0]):
        h = np.array(np.zeros((hidden_dim,1)),dtype=fpt)
        #print(h)
        for t in range(seq_max_len):
            x=np.array((I*(np.array(points[i][slice(t*stride,t*stride+window)])-fpt(mean)))/fpt(std),dtype=fpt).reshape((-1,1))
            pre = np.array((np.matmul(np.transpose(qW2),np.matmul(np.transpose(qW1),x)) + np.matmul(np.transpose(qU2),np.matmul(np.transpose(qU1),h)))/(q*1),dtype=fpt)
            h_ = np.array(nonlin("t",pre+qB_h*I,q*I)/(q),dtype=fpt)
            z = np.array(nonlin("s",pre+qB_g*I,q*I)/(q),dtype=fpt)
            h = np.array((np.multiply(z,h) + np.array(np.multiply(fpt(I*zeta)*(I-z)+fpt(I*nu)*I,h_)/I,dtype=fpt))/I,dtype=fpt)
        
        pred_lbls.append(np.argmax(np.matmul(np.transpose(h),qFC_Weight) + qFC_Bias))
    pred_lbls = np.array(pred_lbls)
    print(lbls)
    print(pred_lbls)
    print(float((pred_lbls==lbls).sum())/lbls.shape[0])

def predict_qfp(points,lbls):
    pred_lbls = []
    
    for i in range(points.shape[0]):
        h = np.array(np.zeros((hidden_dim,1)),dtype=float)
        for t in range(seq_max_len):
            x=np.array(((np.array(points[i][slice(t*stride,t*stride+window)])-float(mean)))/float(std),dtype=float).reshape((-1,1))
            pre = np.array(np.matmul(np.transpose(np.matmul(qW1/q,qW2/q)),x) + np.matmul(np.transpose(np.matmul(qU1/q,qU2/q)),h),dtype=float)
            h_ = np.array(np.maximum(-1,np.minimum(1,pre+qB_h/q)),dtype=float)
            z = np.array(np.maximum(np.minimum(0.5*(pre+qB_g/q+1),1),0),dtype=float)
            h = np.array((np.multiply(z,h) + np.array(np.multiply(zeta*(1-z)+nu,h_),dtype=float)),dtype=float)
        
        pred_lbls.append(np.argmax(np.matmul(np.transpose(h),FC_Weight) + FC_Bias))
    pred_lbls = np.array(pred_lbls)
    print(lbls)
    print(pred_lbls)
    print(float((pred_lbls==lbls).sum())/lbls.shape[0])

predict(test_cuts,test_cuts_lbls)
#predict_fp(test_cuts,test_cuts_lbls)

#predict_qfp(test_cuts,test_cuts_lbls)
