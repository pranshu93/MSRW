from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from math import sqrt
from tensorflow.contrib import rnn
from rnn import FastRNNCell,FastGRNNCell

os.environ["CUDA_VISIBLE_DEVICES"]=""
tf.reset_default_graph()

def cut_acc(cuts,lbls,p_lbls):
    acc = 0; 
    start_index = 0
    ctr = 0        
    for i in range(lbls.shape[0]):
        window_length = int(np.ceil(float(cuts[i].__len__()-window)/(window-overlap)))
        end_index = start_index + window_length 
        if(window_length > 0):
            #print(lbls[i],p_lbls[slice(start_index,end_index)])
            ctr += 1
            sum = np.sum(p_lbls[start_index:end_index])
            acc += (lbls[i] == (sum > float(end_index - start_index) * 0.5))
            #acc += ((1 - lbls[i]) == ((end_index - start_index - sum) * sca < 0.5 * float(end_index - start_index)))
        start_index = end_index
    return float(acc)/ctr
    #return float(acc)/lbls.shape[0] 

def forward_iter(data, labels, index, code):
    batch_lbls = labels[index]
    batchx = data[index].reshape((batch_lbls.size, time_steps, input_dim))
    batchy = np.zeros((batch_lbls.size, num_classes)); batchy[np.arange(batch_lbls.size),batch_lbls] = 1;
    if(code):
        sess.run(train_op, feed_dict={X: batchx, Y: batchy, learning_rate: lr})
    else:        
        return(sess.run([accuracy,pred_labels], feed_dict={X: batchx, Y: batchy}))

def RNN(x):
    x = tf.unstack(x, time_steps, 1)
    lstm_cell = FastGRNNCell(hidden_dim)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=0.25)
    #lstm_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def process(data,labels,window,overlap):
    cr_data = []; cr_labels = [];
    for i in range(data.__len__()):
        num_iter = int(np.ceil(float(data[i].__len__()-window)/(window-overlap)))
        arr = [0] * int((window*num_iter-overlap*(num_iter-1)))
        arr[:data[i].__len__()] = data[i]
        [cr_data.append(arr[slice(int(j*(window-overlap)),int((j+1)*window-j*overlap))]) for j in range(num_iter)]
        cr_labels.extend([labels[i]]*num_iter)
    cr_data = np.array(cr_data); cr_labels = np.array(cr_labels)
    return cr_data, cr_labels

window = 384
stride = window/2 ; overlap = window-stride;

fileloc = "/home/pranshu/Dataset/Activity/"

cv_ind = 5 
train_cuts = []; train_cuts_lbls = [];
for i in range(5):
    if(i != cv_ind - 1):
        cuts = np.load(fileloc + "f" + str(i) + "_cuts.npy"); train_cuts = np.concatenate([train_cuts,cuts]);
        labels = np.load(fileloc + "f" + str(i) + "_cuts_lbls.npy"); train_cuts_lbls = np.concatenate([train_cuts_lbls,labels]);

test_cuts = np.load(fileloc + "f" + str(cv_ind - 1) + "_cuts.npy"); test_cuts_lbls = np.load(fileloc + "f" + str(cv_ind - 1) + "_cuts_lbls.npy")

all_cuts = []; [all_cuts.extend(train_cuts[i]) for i in range(train_cuts.shape[0])];
mean = np.mean(np.array(all_cuts)); std = np.std(np.array(all_cuts));
train_cuts_n = []; test_cuts_n = []; 
[train_cuts_n.append(((np.array(train_cuts[i])-mean)/std).tolist()) for i in range(train_cuts.shape[0])]
[test_cuts_n.append(((np.array(test_cuts[i])-mean)/std).tolist()) for i in range(test_cuts.shape[0])]

train_data, train_labels = process(train_cuts_n,train_cuts_lbls,window,overlap)
test_data, test_labels = process(test_cuts_n,test_cuts_lbls,window,overlap)

#print(float(np.sum(test_labels))/test_labels.shape[0])
#print(float(np.sum(test_cuts_lbls))/test_cuts_lbls.shape[0])
#print(train_data.shape[0])

sca = (1 - float(np.sum(test_cuts_lbls))/test_cuts_lbls.shape[0])/(1 - float(np.sum(test_labels))/test_labels.shape[0])
train_labels = np.array(train_labels - np.min(train_labels),dtype=int)
test_labels = np.array(test_labels - np.min(test_labels),dtype=int)

lr = 0.002

num_epochs = 100 
batch_size = 96 

input_dim = 32 
time_steps = int(train_data[0].__len__()/input_dim)
hidden_dim = 20 
num_classes = np.max(train_labels) - np.min(train_labels) + 1

X = tf.placeholder("float", [None, time_steps, input_dim])
Y = tf.placeholder("float", [None, num_classes])

learning_rate = tf.placeholder("float", shape=(), name='learning_rate')
tf.set_random_seed(42)
np.random.seed(42)

weights = {
    'out': tf.Variable(tf.random_normal([hidden_dim, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = RNN(X)
prediction = tf.nn.softmax(logits)
pred_labels = tf.argmax(prediction, 1)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate) 
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True) 
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(pred_labels, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))

max_acc = 0; best_iter = 0;
for i in range(num_epochs):
    if(i % 150 == 149):
        lr = lr/2
    num_iter = int(train_data.shape[0]/batch_size)
    [forward_iter(train_data,train_labels,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(train_data,train_labels,slice(num_iter*batch_size,train_data.shape[0]),True)

    tr_acc, _ = forward_iter(train_data,train_labels,slice(0,train_data.shape[0]),False)
    acc, p_lbls = forward_iter(test_data,test_labels,slice(0,test_data.shape[0]),False)
    #print(test_cuts_lbls[i])
    #print(p_lbls[slice(0,int(np.ceil(float(cuts[0].__len__()-window)/(window-overlap))))])
    acc1 = cut_acc(test_cuts,test_cuts_lbls,p_lbls)
    if(max_acc < acc1):
        max_acc = acc1
        best_iter = i
    print(i,tr_acc,acc,acc1)
print(max_acc)


