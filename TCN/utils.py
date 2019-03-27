import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
import argparse
import sys

def multiClassHingeLoss(logits, label, batch_th):
    '''
    MultiClassHingeLoss to match C++ Version - No TF internal version
    '''
    flatLogits = tf.reshape(logits, [-1, ])
    correctId = tf.range(0, batch_th) * logits.shape[1] + label
    correctLogit = tf.gather(flatLogits, correctId)

    maxLabel = tf.argmax(logits, 1)
    top2, _ = tf.nn.top_k(logits, k=2, sorted=True)

    wrongMaxLogit = tf.where(tf.equal(maxLabel, label), top2[:, 1], top2[:, 0])

    return tf.reduce_mean(tf.nn.relu(1. + wrongMaxLogit - correctLogit))


def crossEntropyLoss(logits, label):
    '''
    Cross Entropy loss for MultiClass case in joint training for
    faster convergence
    '''
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))


def hardThreshold(A, s):
    '''
    Hard Thresholding function on Tensor A with sparsity s
    '''
    A_ = np.copy(A)
    A_ = A_.ravel()
    if len(A_) > 0:
        th = np.percentile(np.abs(A_), (1 - s) * 100.0, interpolation='higher')
        A_[np.abs(A_) < th] = 0.0
    A_ = A_.reshape(A.shape)
    return A_


def copySupport(src, dest):
    '''
    copy support of src tensor to dest tensor
    '''
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest


def countnnZ(A, s, bytesPerVar=4):
    '''
    Returns # of nonzeros and represnetative size of the tensor
    Uses dense for s >= 0.5 - 4 byte
    Else uses sparse - 8 byte
    '''
    params = 1
    hasSparse = False
    for i in range(0, len(A.shape)):
        params *= int(A.shape[i])
    if s < 0.5:
        nnZ = np.ceil(params * s)
        hasSparse = True
        return nnZ, nnZ * 2 * bytesPerVar, hasSparse
    else:
        nnZ = params
        return nnZ, nnZ * bytesPerVar, hasSparse


def gen_non_linearity(A, non_linearity):
    '''
    Returns required activation for a tensor based on the inputs
    '''
    if non_linearity == "tanh":
        return math_ops.tanh(A)
    elif non_linearity == "sigmoid":
        return math_ops.sigmoid(A)
    elif non_linearity == "relu":
        return gen_math_ops.maximum(A, 0.0)
    elif non_linearity == "quantTanh":
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), -1.0)
    elif non_linearity == "quantSigm":
        A = (A + 1.0) / 2.0
        return gen_math_ops.maximum(gen_math_ops.minimum(A, 1.0), 0.0)
    else:
        return math_ops.tanh(A)


# Auxiliary methods for EMI-RNN
# Will probably be moved out 

def getConfusionMatrix(predicted, target, numClasses):
    '''
    confusion[i][j]: Number of elements of class j
        predicted as class i
    '''
    assert(predicted.ndim == 1)
    assert(target.ndim == 1)
    arr = np.zeros([numClasses, numClasses])
    
    for i in range(len(predicted)):
        arr[predicted[i]][target[i]] += 1
    return arr

def printFormattedConfusionMatrix(matrix):
    '''
    Given a 2D confusion matrix, prints it in a formatte
    way
    '''
    assert(matrix.ndim == 2)
    assert(matrix.shape[0] == matrix.shape[1])
    RECALL = 'Recall'
    PRECISION = 'PRECISION'
    print("|%s|"% ('True->'), end='')
    for i in range(matrix.shape[0]):
        print("%7d|" % i, end='')
    print("%s|" % 'Precision')
    
    print("|%s|"% ('-'* len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-'* 7), end='')
    print("%s|" % ('-'* len(PRECISION)))
    
    precisionlist = np.sum(matrix, axis=1)
    recalllist = np.sum(matrix, axis=0) 
    precisionlist = [matrix[i][i]/ x if x != 0 else -1 for i,x in enumerate(precisionlist)]
    recalllist = [matrix[i][i]/x if x != 0 else -1for i,x in enumerate(recalllist)]
    for i in range(matrix.shape[0]):
        # len recall = 6
        print("|%6d|"% (i), end='')
        for j in range(matrix.shape[0]):
            print("%7d|" % (matrix[i][j]), end='')
        print("%s" % (" " * (len(PRECISION) - 7)), end='')
        if precisionlist[i] != -1:
            print("%1.5f|" % precisionlist[i])
        else:
            print("%7s|" % "nan")
    
    print("|%s|"% ('-'* len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-'* 7), end='')
    print("%s|" % ('-'*len(PRECISION)))
    print("|%s|"% ('Recall'), end='')
    
    for i in range(matrix.shape[0]):
        if recalllist[i] != -1:
            print("%1.5f|" % (recalllist[i]), end='')
        else:
            print("%7s|" % "nan", end='')

    print('%s|' % (' ' * len(PRECISION)))    
    
    
def getPrecisionRecall(cmatrix, label=1):
    trueP = cmatrix[label][label]
    denom = np.sum(cmatrix, axis=0)[label]
    if denom == 0:
        denom = 1
    recall = trueP / denom
    denom = np.sum(cmatrix, axis=1)[label]
    if denom == 0:
        denom = 1
    precision = trueP / denom
    return precision, recall


def getMacroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0)
    precisionlist__ = [cmatrix[i][i]/ x if x!= 0 else 0 for i,x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i]/x if x!=0 else 0 for i,x in enumerate(recalllist)]
    precision = np.sum(precisionlist__)
    precision /= len(precisionlist__)
    recall = np.sum(recalllist__)
    recall /= len(recalllist__)
    return precision, recall


def getMicroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0) 
    num =0.0
    for i in range(len(cmatrix)):
        num += cmatrix[i][i]

    precision = num / np.sum(precisionlist)
    recall = num / np.sum(recalllist)
    return precision, recall


def getMacroMicroFScore(cmatrix):
    '''
    Returns macro and micro f-scores.
    Refer: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
    '''
    precisionlist = np.sum(cmatrix, axis=1)
    recalllist = np.sum(cmatrix, axis=0) 
    precisionlist__ = [cmatrix[i][i]/ x if x != 0 else 0 for i,x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i]/x if x != 0 else 0 for i,x in enumerate(recalllist)]
    macro = 0.0
    for i in range(len(precisionlist)):
        denom = precisionlist__[i] + recalllist__[i]
        numer = precisionlist__[i] * recalllist__[i] * 2
        if denom == 0:
            denom = 1
        macro += numer / denom
    macro /= len(precisionlist)
    
    num = 0.0
    for i in range(len(precisionlist)):
        num += cmatrix[i][i]

    denom1 = np.sum(precisionlist)
    denom2 = np.sum(recalllist)
    pi = num / denom1
    rho = num / denom2
    denom = pi + rho
    if denom == 0:
        denom = 1
    micro = 2 * pi * rho / denom
    return macro, micro

def getArgs():
    parser = argparse.ArgumentParser(description='HyperParameters for Dynamic RNN Algorithm')
    parser.add_argument('-ct', type=bool, default=True, help='FastRNN(False)/FastGRNN(True)')
    parser.add_argument('-dt', type=bool, default=False, help='Fully Connected Layer/Bonsai')
    parser.add_argument('-hP', type=bool, default=True, help='Have Projection(False/True)')
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
    parser.add_argument('-ot', type=int, default=0, help='Adam(0)/Momentum(1)')
    parser.add_argument('-ml', type=int, default=768, help='Maximum slice length of cut taken for classification')
    parser.add_argument('-nc', type=int, default=2, help='Number of classes')
    parser.add_argument('-fn', type=int, default=3, help='Fold Number to classify for cross validation[1/2/3/4/5]')
    parser.add_argument('-ne', type=int, default=500, help='Number of epochs to run training routine for')
    parser.add_argument('-q15', type=bool, default=False, help='Represent input as Q15?')
    parser.add_argument('-out', type=str, default=sys.stdout, help='Output filename')
    parser.add_argument('-type', type=str, default='tar', help='Classification type: \'tar\' for target,' \
                                                               ' \'act\' for activity)')
    parser.add_argument('-sig', type=float, default=4.0, help='Sigma parameter in Bonsai')
    parser.add_argument('-dep', type=int, default=2, help='Depth of Bonsai tree')
    parser.add_argument('-P', type=int, default=5, help='Bonsai projection dimension')
    parser.add_argument('-rZ', type=float, default=1.0, help='Bonsai regularization Z')
    parser.add_argument('-rT', type=float, default=1.0, help='Bonsai regularization Theta')
    parser.add_argument('-rW', type=float, default=1.0, help='Bonsai regularization W')
    parser.add_argument('-rV', type=float, default=1.0, help='Bonsai regularization V')
    parser.add_argument('-sZ', type=float, default=1.0, help='Bonsai sparsity Z')
    parser.add_argument('-sT', type=float, default=1.0, help='Bonsai sparsity T')
    parser.add_argument('-sW', type=float, default=1.0, help='Bonsai sparsity W')
    parser.add_argument('-sV', type=float, default=1.0, help='Bonsai sparsity V')
    parser.add_argument('-base', type=str, default='/fs/project/PAS1090/radar/Austere/Bora_New_Detector', help='Base location of data')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout applied to layers (default: 0.0)')
    parser.add_argument('-clip', type=float, default=1, help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('-ksize', type=int, default=16,help='kernel size (default: 7)')
    parser.add_argument('-levels', type=int, default=2,help='# of levels (default: 8)')
    parser.add_argument('-nhid', type=int, default=16,help='number of hidden units per layer (default: 25)')
    return parser.parse_args()

'''
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
    parser.add_argument('-lr', type=float, default=0.005, help='Learning Rate of Optimisation')
    parser.add_argument('-bs', type=int, default=128, help='Batch Size of Optimisation')
    parser.add_argument('-hs', type=int, default=16, help='Hidden Layer Size')
    parser.add_argument('-ot', type=int, default=1, help='Adam(False)/Momentum(True)')
    parser.add_argument('-ml', type=int, default=768, help='Maximum slice length of cut taken for classification')
    parser.add_argument('-fn', type=int, default=3, help='Fold Number to classify for cross validation[1/2/3/4/5]')
    parser.add_argument('-ne', type=int, default=500, help='Number of epochs to run training routine for')
    return parser.parse_args()
'''

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

'''
variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)
for k, v in zip(variables_names, values):
    if(k.find("Variable:0") != -1):        
        np.save(modelloc + '/' + "FC_Weight",v)
    if(k.find("Variable_1:0") != -1):        
        np.save(modelloc + '/' + "FC_Bias",v)
    if(k.find("W1") != -1):        
        np.save(modelloc + '/' + "W1",v)
    if(k.find("W2") != -1):        
        np.save(modelloc + '/' + "W2",v)
    if(k.find("U1") != -1):        
        np.save(modelloc + '/' + "U1",v)
    if(k.find("U2") != -1):        
        np.save(modelloc + '/' + "U2",v)
    if(k.find("zeta") != -1):        
        np.save(modelloc + '/' + "zeta",v)
    if(k.find("nu") != -1):        
        np.save(modelloc + '/' + "nu",v)
    if(k.find("B_g") != -1):        
        np.save(modelloc + '/' + "B_g",v)
    if(k.find("B_h") != -1):        
        np.save(modelloc + '/' + "B_h",v)

os.system("python3 " + os.path.abspath('../../EdgeML/tf/examples/FastCells/quantizeFastModels.py') + " -dir " + modelloc)

qW1 = np.load(modelloc + "/QuantizedFastModel/qW1.npy") 
qFC_Bias = np.load(modelloc + "/QuantizedFastModel/qFC_Bias.npy") 
qW2 = np.load(modelloc + "/QuantizedFastModel/qW2.npy") 
qU2 = np.load(modelloc + "/QuantizedFastModel/qU2.npy") 
qFC_Weight = np.load(modelloc + "/QuantizedFastModel/qFC_Weight.npy") 
qU1 = np.load(modelloc + "/QuantizedFastModel/qU1.npy") 
qB_g = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_g.npy")) 
qB_h = np.transpose(np.load(modelloc + "/QuantizedFastModel/qB_h.npy"))
q = np.load(modelloc + "/QuantizedFastModel/paramScaleFactor.npy")

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
#np.save(modelloc + '/' + k.replace('/','_'),v)
#print ("Variable: ", k)
#print ("Shape: ", v.shape)
#print (formatp(v))
    predict(test_cuts,test_cuts_lbls)
    '''
''''''
