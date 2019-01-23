'''
Copied from Microsoft's EdgeML repository
'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def multi_class_hinge_loss(logits, label, batch_th):
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


def cross_entropy_loss(logits, label):
    '''
    Cross Entropy loss for MultiClass case in joint training for faster convergence
    '''
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))


def hard_thrsd(A, s):
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


def copy_support(src, dest):
    '''
    copy support of src tensor to dest tensor
    '''
    support = np.nonzero(src)
    dest_ = dest
    dest = np.zeros(dest_.shape)
    dest[support] = dest_[support]
    return dest


def count_nnZ(A, s, bytesPerVar = 4):
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


# Added by Dhrubo
def train_test_partition(data, labels, test_size, seed):
    '''
    Creates train-test partitioning of input file
    :param data: Input data
    :param labels: Input labels
    :param test_size: Percentage of split allocated to test
    :param seed: Random seed (for reproducibility)
    :return: data_train, data_test, labels_train, labels_test
    '''
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=seed)
    return data_train, data_test, labels_train, labels_test
