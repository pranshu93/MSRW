import numpy as np
import tensorflow as tf
import time
from tcn import *
import os
from utils import *

np.random.seed(42); tf.set_random_seed(42);

args = getArgs()

def forward_iter(data, label, index, code):
    batchx = data[index];  batchy = label[index]; 
    if(code): sess.run(update_step, feed_dict={inputs: batchx, labels: batchy,  learning_rate: lr})
    else: return(sess.run(accuracy, feed_dict={inputs: batchx, labels: batchy,  learning_rate: lr}))

def TCN(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    # tcn is of shape (batch_size, seq_len, num_channels[-1](usually hidden size))
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    linear = tf.contrib.layers.fully_connected(tcn[:, -1, :], output_size, activation_fn=None)
    return linear

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

#fileloc = os.path.abspath('../Datasets/Radar2/')
fileloc = args.base

#tryloc = os.path.abspath('../Datasets/Austere/')
#modelloc = os.path.abspath('../Models/')

#modelloc = "/scratch/cse/phd/anz178419/Models/MSRW/"

cv_ind = args.fn
train_cuts = []; train_cuts_lbls = [];
for i in range(5):
    if(i != cv_ind - 1):
        cuts = np.load(fileloc + "/" + args.type + str(i) + "_cuts.npy"); train_cuts = np.concatenate([train_cuts,cuts]);
        labels = np.load(fileloc + "/" + args.type + str(i) + "_cuts_lbls.npy"); train_cuts_lbls = np.concatenate([train_cuts_lbls,labels]);

test_cuts = np.load(fileloc + "/" + args.type + str(cv_ind - 1) + "_cuts.npy")
test_cuts_lbls = np.load(fileloc + "/" + args.type + str(cv_ind - 1) + "_cuts_lbls.npy")

max_length = args.ml

#seq_max_len = int(np.floor(float(max_length-window)/stride)+1)
seq_max_len = int(np.ceil(float(max_length-window)/stride))

all_cuts = []; [all_cuts.extend(train_cuts[i]) for i in range(train_cuts.shape[0])];
mean = np.mean(np.array(all_cuts)); std = np.std(np.array(all_cuts));

train_cuts_n = []; test_cuts_n = []; try_cuts_n = [];
[train_cuts_n.append(((np.array(train_cuts[i])-mean)/std).tolist()) for i in range(train_cuts.shape[0])]
[test_cuts_n.append(((np.array(test_cuts[i])-mean)/std).tolist()) for i in range(test_cuts.shape[0])]

lr = args.lr

num_epochs = args.ne
batch_size = args.bs

hidden_dim = args.hs
num_classes = 2

batch_size = args.bs
n_classes = num_classes

train_data, train_labels, train_seqlen = process(train_cuts_n,train_cuts_lbls)
test_data, test_labels, test_seqlen = process(test_cuts_n,test_cuts_lbls)

X_train = np.array(train_data); X_test = np.array(test_data);
Y_train = np.array(train_labels); Y_test = np.array(test_labels);

in_channels = X_train.shape[2]
seq_length = X_train.shape[1]

labels = tf.placeholder(tf.float32, (None, n_classes))
inputs = tf.placeholder(tf.float32, (None, seq_length, in_channels))

channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
outputs = TCN(inputs, n_classes, channel_sizes, seq_length, kernel_size=kernel_size, dropout=dropout)
predictions = tf.argmax(outputs, axis=-1)

correct_pred = tf.equal(predictions, tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= outputs, labels=labels)

lr = args.lr
learning_rate = tf.placeholder(tf.float32, shape=[])

if(args.ot==1):
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True) 
else:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gradients, variables = zip(*optimizer.compute_gradients(loss))
if args.clip > 0:
    gradients, _ = tf.clip_by_global_norm(gradients, args.clip)
update_step = optimizer.apply_gradients(zip(gradients, variables))

sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables()))) 

saver = tf.train.Saver()

max_acc = 0
for i in range(num_epochs):
    num_iter = int(X_train.shape[0]/batch_size)
    [forward_iter(X_train,Y_train,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(X_train,Y_train,slice(num_iter*batch_size,X_train.shape[0]),True)

    tr_acc = forward_iter(X_train,Y_train,slice(0,test_data.__len__()),False)
    acc = forward_iter(X_test,Y_test,slice(0,train_data.__len__()),False)

    if(max_acc < acc):
        max_acc = acc
    #    saver.save(sess, modelloc + "/bestmodel.ckpt")	
    print(i,tr_acc,acc)
print(max_acc)

# Create result string
results_list = [args.ksize, args.clip, args.levels, args.nhid, args.w, args.ml, args.dropout, args.sp, args.lr,
                args.ne, args.bs, args.hs, args.ot, args.fn, max_acc]

# Print to output file
out_handle = open(args.out, "a")
# Write a line of output
out_handle.write('\t'.join(map(str, results_list)) + '\n')
out_handle.close()
