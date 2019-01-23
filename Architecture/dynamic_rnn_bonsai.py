from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import os
import argparse
from math import sqrt
from tensorflow.contrib import rnn
from rnn import FastRNNCell,FastGRNNCell
from utils import utils
import sys

#os.environ["CUDA_VISIBLE_DEVICES"]=""

# Making sure MSRW is part of python path
sys.path.insert(0, '../')
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

np.random.seed(42)
tf.set_random_seed(42)

class Bonsai:
    def __init__(self, x, C, F, P, D, S, lW, lT, lV, lZ,
                 sW, sT, sV, sZ, lr=None, W=None, T=None, V=None, Z=None):

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
            self.margin_loss = utils.multi_class_hinge_loss(tf.transpose(self.score), tf.argmax(self.y, 1), self.batch_th)
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

    '''def analyse(self):
        _feed_dict = {self.x: train_feats, self.y: train_labels}
        x_cap_eval = self.X_eval.eval(feed_dict=_feed_dict)
        tt1 = self.T_eval.eval()
        prob = []
        for i in range(0, self.internalNodes):
            prob.append(np.dot(tt1[i], x_cap_eval))
        prob = np.array(prob)
        nodes = np.zeros(self.internalNodes + 1)
        for i in range(x_cap_eval.shape[1]):
            j = 0
            while j < self.internalNodes:
                if (prob[j][i] > 0):
                    if (2 * j + 1 < self.internalNodes):
                        j = 2 * j + 1
                    else:
                        j = 2 * j + 1
                        nodes[j - self.internalNodes] = nodes[j - self.internalNodes] + 1
                else:
                    if (2 * j + 2 < self.internalNodes):
                        j = 2 * j + 2
                    else:
                        j = 2 * j + 2
                        nodes[j - self.internalNodes] = nodes[j - self.internalNodes] + 1
        for i in range(0, self.internalNodes + 1):
            print(i, nodes[i])'''

def getArgs():
    parser = argparse.ArgumentParser(description='HyperParameters for Dynamic RNN Algorithm')
    parser.add_argument('-ct', type=int, default=1, help='FastRNN(False)/FastGRNN(True)')
    parser.add_argument('-unl', type=str, default="tanh" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
    parser.add_argument('-gunl', type=str, default="tanh" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
    parser.add_argument('-ggnl', type=str, default="sigmoid" , help='tanh/sigmoid/relu/quantSigm/quantTanh')
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
    parser.add_argument('-q15', type=bool, default=False, help='Represent input as Q15?')
    parser.add_argument('-out', type=str, default=sys.stdout, help='Output filename')
    parser.add_argument('-type', type=str, default='tar', help='Classification type: \'tar\' for target,' \
                                                               ' \'act\' for activity)')
    parser.add_argument('-base', type=str, default='/fs/project/PAS1090/radar/Austere/Bora_New_Detector/',
                        help='Base location of data')
    return parser.parse_args()

def q15_to_float(arr):
    return arr*100000.0/32768.0

'''def forward_iter(data, labels, data_seqlen, index, code):
    batchx = data[index];  batchy = labels[index]; batchz = data_seqlen[index]
    if(code): sess.run(train_op, feed_dict={X: batchx, Y:batchy, seqlen: batchz, learning_rate: lr})
    else: return(sess.run(accuracy, feed_dict={X: batchx, Y: batchy, seqlen: batchz, learning_rate: lr}))'''

def dynamicRNNFeaturizer(x):
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
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, hidden_dim]), index)
    return outputs

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


#### Main function ####
args = getArgs()
window = args.w
stride = int(window * args.sp);

# Get Bonsai values
sigma = args.sigma
depth = args.depth

projectionDimension = args.proj_dim
regZ = args.rZ
regT = args.rT
regW = args.rW
regV = args.rV

totalEpochs = args.epochs

learningRate = args.learning_rate

dataDimension = args.F
numClasses = args.nc

sparZ = args.sZ

if numClasses > 2:
    sparW = 0.2
    sparV = 0.2
    sparT = 0.2
else:
    sparW = 1
    sparV = 1
    sparT = 1

if args.sW is not None:
    sparW = args.sW
if args.sV is not None:
    sparV = args.sV
if args.sT is not None:
    sparT = args.sT

useMCHLoss = True

if numClasses == 2:
    numClasses = 1

'''if args.bat=='pbs':
    fileloc = os.path.abspath('/fs/project/PAS1090/radar/Austere/Bora_New_Detector/')
elif args.bat=='slurm':
    fileloc = os.path.abspath('/scratch/dr2915/Austere/Bora_New_Detector/')
else:
    raise NotImplementedError'''

fileloc = os.path.abspath(args.base)
#tryloc = os.path.abspath('/home/cse/phd/anz178419/MSRW/Datasets/Austere/')
#modelloc = "/scratch/cse/phd/anz178419/Models/MSRW/"

cv_ind = args.fn
train_cuts = []; train_cuts_lbls = [];
for i in range(5):
    if(i != cv_ind - 1):
        cuts = np.load(fileloc + "/" + args.type + str(i) + "_cuts.npy"); train_cuts = np.concatenate([train_cuts,cuts]);
        labels = np.load(fileloc + "/" + args.type + str(i) + "_cuts_lbls.npy"); train_cuts_lbls = np.concatenate([train_cuts_lbls,labels]);

test_cuts = np.load(fileloc + "/" + args.type + str(cv_ind - 1) + "_cuts.npy"); test_cuts_lbls = np.load(fileloc + "/" + args.type + str(cv_ind - 1) + "_cuts_lbls.npy")
#try_cuts = np.load(tryloc + "/f_cuts.npy"); try_cuts_lbls = np.load(tryloc + "/f_cuts_lbls.npy")


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
#print(mean,std)
train_cuts_n = []; test_cuts_n = []; try_cuts_n = [];

# Are we representing input as Q15?
if args.q15:
    mean = int(mean)
    std = int(std)
    [train_cuts_n.append(q15_to_float(((np.array(train_cuts[i])-mean)/std).astype(int)).tolist()) for i in range(train_cuts.shape[0])]
    [test_cuts_n.append(q15_to_float(((np.array(test_cuts[i])-mean)/std).astype(int)).tolist()) for i in range(test_cuts.shape[0])]
else:
    [train_cuts_n.append(((np.array(train_cuts[i]) - mean) / std).tolist()) for i in range(train_cuts.shape[0])]
    [test_cuts_n.append(((np.array(test_cuts[i]) - mean) / std).tolist()) for i in range(test_cuts.shape[0])]
#[try_cuts_n.append(((np.array(try_cuts[i])-mean)/std).tolist()) for i in range(try_cuts.shape[0])]

lr = args.lr

num_epochs = 500
batch_size = args.bs

hidden_dim = args.hs
num_classes = 2

train_feats, train_labels, train_seqlen = process(train_cuts_n,train_cuts_lbls)
test_feats, test_labels, test_seqlen = process(test_cuts_n,test_cuts_lbls)
#try_data, try_labels, try_seqlen = process(try_cuts_n,try_cuts_lbls)

X = tf.placeholder("float", [None, seq_max_len, window])
#Y = tf.placeholder("float", [None, num_classes])

learning_rate = tf.placeholder("float", shape=(), name='learning_rate')
tf.set_random_seed(42)
np.random.seed(42)

seqlen = tf.placeholder(tf.int32, [None])

features = dynamicRNNFeaturizer(X)
# Connect Bonsai graph
bonsaiObj = Bonsai(dynamicRNNFeaturizer(X), numClasses, dataDimension,
                       projectionDimension, depth, sigma)

###### COMMENTED FROM PRANSHU'S CODE ######
#logits = bonsaiObj.bonsaiGraph(features)
#prediction = tf.nn.softmax(logits)
#pred_labels = tf.argmax(prediction, 1)

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#if(args.ot):
#    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True)
#else:
#    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#train_op = optimizer.minimize(loss_op)

#correct_pred = tf.equal(pred_labels, tf.argmax(Y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
###### COMMENTED FROM PRANSHU'S CODE ######

#sess = tf.InteractiveSession(config=config)
tf.reset_default_graph()
sess = tf.InteractiveSession()
sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_variables(tf.local_variables())))

saver = tf.train.Saver()

max_acc = 0; max_try_acc=0; best_iter = 0

num_iters = int(train_feats.__len__()/batch_size)
total_batches = num_iters * num_epochs

counter = 0
if bonsaiObj.numClasses > 2:
    trimlevel = 15
else:
    trimlevel = 5
iht_done = 0

for i in range(num_epochs):
    accu = 0.0
    for j in range(num_iters):

        if ((counter == 0) or (counter == total_batches / 3) or (counter == 2 * total_batches / 3)):
            bonsaiObj.sigmaI = 1
            iters_phase = 0

        elif (iters_phase % 100 == 0):
            indices = np.random.choice(train_feats.shape[0], 100)
            batch_x = train_feats[indices, :]
            batch_y = train_labels[indices, :]
            batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
            batch_z = train_seqlen[indices, :]

            _feed_dict = {X: batch_x, bonsaiObj.y: batch_y, seqlen: batch_z}
            x_cap_eval = bonsaiObj.X_eval.eval(feed_dict=_feed_dict)
            T_eval = bonsaiObj.T_eval.eval()
            sum_tr = 0.0
            for k in range(0, bonsaiObj.internalNodes):
                sum_tr = sum_tr + (np.sum(np.abs(np.dot(T_eval[k], x_cap_eval))))

            if (bonsaiObj.internalNodes > 0):
                sum_tr = sum_tr / (100 * bonsaiObj.internalNodes)
                sum_tr = 0.1 / sum_tr
            else:
                sum_tr = 0.1
            sum_tr = min(1000, sum_tr * (2 ** (float(iters_phase) / (float(total_batches) / 30.0))))

            bonsaiObj.sigmaI = sum_tr

        iters_phase = iters_phase + 1
        batch_x = train_feats[j * batch_size:(j + 1) * batch_size]
        batch_y = train_labels[j * batch_size:(j + 1) * batch_size]
        batch_y = np.reshape(batch_y, [-1, bonsaiObj.numClasses])
        batch_z = train_seqlen[j * batch_size:(j + 1) * batch_size]

        if bonsaiObj.numClasses > 2:
            _feed_dict = {X: batch_x, bonsaiObj.y: batch_y, seqlen: batch_z, bonsaiObj.batch_th: batch_y.shape[0]}
        else:
            _feed_dict = {X: batch_x, bonsaiObj.y: batch_y, seqlen: batch_z}

        _, loss1 = sess.run([bonsaiObj.train_stepW, bonsaiObj.loss], feed_dict=_feed_dict)
        _, loss1 = sess.run([bonsaiObj.train_stepV, bonsaiObj.loss], feed_dict=_feed_dict)
        _, loss1 = sess.run([bonsaiObj.train_stepT, bonsaiObj.loss], feed_dict=_feed_dict)
        _, loss1 = sess.run([bonsaiObj.train_stepZ, bonsaiObj.loss], feed_dict=_feed_dict)
        temp = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
        accu = temp + accu

        if counter >= total_batches / 3 and counter < 2 * total_batches / 3:
            if counter % trimlevel == 0:
                W_old = bonsaiObj.W_eval.eval()
                V_old = bonsaiObj.V_eval.eval()
                Z_old = bonsaiObj.Z_eval.eval()
                T_old = bonsaiObj.T_eval.eval()

                W_new = utils.hard_thrsd(W_old, bonsaiObj.sW)
                V_new = utils.hard_thrsd(V_old, bonsaiObj.sV)
                Z_new = utils.hard_thrsd(Z_old, bonsaiObj.sZ)
                T_new = utils.hard_thrsd(T_old, bonsaiObj.sT)

                if counter % num_iters == 0:
                    print("IHT", np.count_nonzero(W_new), np.count_nonzero(V_new), np.count_nonzero(Z_new),
                          np.count_nonzero(T_new))

                fd_thrsd = {bonsaiObj.W_th: W_new, bonsaiObj.V_th: V_new, bonsaiObj.Z_th: Z_new, bonsaiObj.T_th: T_new}
                sess.run(bonsaiObj.hard_thrsd_grp, feed_dict=fd_thrsd)

                iht_done = 1
            elif ((iht_done == 1 and counter >= (total_batches / 3) and (
                counter < 2 * total_batches / 3) and counter % trimlevel != 0) or (counter >= 2 * total_batches / 3)):
                W_old = bonsaiObj.W_eval.eval()
                V_old = bonsaiObj.V_eval.eval()
                Z_old = bonsaiObj.Z_eval.eval()
                T_old = bonsaiObj.T_eval.eval()

                W_new1 = utils.copy_support(W_new, W_old)
                V_new1 = utils.copy_support(V_new, V_old)
                Z_new1 = utils.copy_support(Z_new, Z_old)
                T_new1 = utils.copy_support(T_new, T_old)

                if counter % num_iters == 0:
                    print("ST", np.count_nonzero(W_new1), np.count_nonzero(V_new1), np.count_nonzero(Z_new1),
                          np.count_nonzero(T_new1))
                    print(8.0 * (
                    np.count_nonzero(W_new) + np.count_nonzero(V_new) + np.count_nonzero(Z_new) + np.count_nonzero(
                        T_new)) / 1024.0)
                fd_st = {bonsaiObj.W_st: W_new1, bonsaiObj.V_st: V_new1, bonsaiObj.Z_st: Z_new1, bonsaiObj.T_st: T_new1}
                sess.run(bonsaiObj.sparse_retrain_grp, feed_dict=fd_st)
        elif ((iht_done == 1 and counter >= (total_batches / 3) and (
            counter < 2 * total_batches / 3) and counter % trimlevel != 0) or (counter >= 2 * total_batches / 3)):
            W_old = bonsaiObj.W_eval.eval()
            V_old = bonsaiObj.V_eval.eval()
            Z_old = bonsaiObj.Z_eval.eval()
            T_old = bonsaiObj.T_eval.eval()

            W_new1 = utils.copy_support(W_new, W_old)
            V_new1 = utils.copy_support(V_new, V_old)
            Z_new1 = utils.copy_support(Z_new, Z_old)
            T_new1 = utils.copy_support(T_new, T_old)

            if counter % num_iters == 0:
                print("ST", np.count_nonzero(W_new1), np.count_nonzero(V_new1), np.count_nonzero(Z_new1),
                      np.count_nonzero(T_new1))
                print(8.0 * (
                np.count_nonzero(W_new) + np.count_nonzero(V_new) + np.count_nonzero(Z_new) + np.count_nonzero(
                    T_new)) / 1024.0)
            fd_st = {bonsaiObj.W_st: W_new1, bonsaiObj.V_st: V_new1, bonsaiObj.Z_st: Z_new1, bonsaiObj.T_st: T_new1}
            sess.run(bonsaiObj.sparse_retrain_grp, feed_dict=fd_st)

        counter = counter + 1

    print("Train accuracy " + str(accu / num_iters))
    #bonsaiObj.analyse()
    if bonsaiObj.numClasses > 2:
        _feed_dict = {X: test_feats, bonsaiObj.y: test_labels, seqlen: test_seqlen, bonsaiObj.batch_th: test_labels.shape[0]}
    else:
        _feed_dict = {X: test_feats, bonsaiObj.y: test_labels, seqlen: test_seqlen}

    old = bonsaiObj.sigmaI
    bonsaiObj.sigmaI = 1e9

    test_acc = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
    print("Test accuracy %g" % test_acc)
    #if bonsaiObj.numClasses > 2:
    #    _feed_dict = {bonsaiObj.x: train_feats, bonsaiObj.y: train_labels, bonsaiObj.batch_th: train_labels.shape[0]}
    #else:
    #    _feed_dict = {bonsaiObj.x: train_feats, bonsaiObj.y: train_labels}
    if bonsaiObj.numClasses > 2:
        _feed_dict = {X: train_feats, bonsaiObj.y: train_labels, seqlen: train_seqlen, bonsaiObj.batch_th: train_labels.shape[0]}
    else:
        _feed_dict = {X: train_feats, bonsaiObj.y: train_labels, seqlen: train_seqlen}

    loss_new = bonsaiObj.loss.eval(feed_dict=_feed_dict)
    reg_loss_new = bonsaiObj.reg_loss.eval(feed_dict=_feed_dict)
    print("Loss %g" % loss_new)
    print("Reg Loss %g" % reg_loss_new)
    train_acc = bonsaiObj.accuracy.eval(feed_dict=_feed_dict)
    print("Train accuracy %g" % train_acc)

    bonsaiObj.sigmaI = old
    print(old)

    print("\n Epoch Number: " + str(i + 1))

'''
for i in range(num_epochs):
    num_iter = int(train_data.__len__()/batch_size)
    [forward_iter(train_data,train_labels,train_seqlen,slice(j*batch_size,(j+1)*batch_size),True) for j in range(num_iter)]
    forward_iter(train_data,train_labels,train_seqlen,slice(num_iter*batch_size,train_data.__len__()),True)
    acc = forward_iter(test_data,test_labels,test_seqlen,slice(0,test_data.__len__()),False)

    tr_acc = forward_iter(train_data,train_labels,train_seqlen,slice(0,train_data.__len__()),False)
    #try_acc = forward_iter(try_data,try_labels,try_seqlen,slice(0,try_data.__len__()),False)

    if(max_acc < acc):	max_acc = acc
    #if(max_try_acc < try_acc):	max_try_acc = try_acc

        #saver.save(sess, modelloc + "bestmodel.ckpt")

        #best_iter = i
    #print(i,tr_acc,acc,try_acc)
print(max_acc,max_try_acc)
'''

# Create result string
results_list = [args.ggnl, args.gunl, args.ur, args.wr, args.w, args.sp, args.lr, args.bs, args.hs, args.ot,
       args.ml, args.fn, max_acc]

# Print to output file
out_handle = open(args.out, "a")
# Write a line of output
out_handle.write('\t'.join(map(str, results_list)) + '\n')
out_handle.close()
