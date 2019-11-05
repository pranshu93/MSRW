import sys
sys.path.insert(0, './')
from Architecture.utils import getConfusionMatrix, printFormattedConfusionMatrix, getPrecisionRecall

import numpy as np

I = 5
num_classes = 2

NUM_HIDDEN = 64
UPDATE_NL = "quantTanh"
GATE_NL = "quantSigm"

test_data = np.load('C++/pedbike_upper_test_3D.npy')
test_lbls = np.load('C++/pedbike_upper_test_lbls.npy')

# Load quantized params
modelloc = 'Models'

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

zeta = 1 / (1 + np.exp(-zeta))
# print("zeta = ",zeta)
nu = np.load(modelloc + "/nu.npy")
nu = 1 / (1 + np.exp(-nu))

# Get mean and std
mean = np.load(modelloc + "/mean.npy")
std = np.load(modelloc + "/std.npy")

def quantTanh(x, scale):
    return np.maximum(-scale, np.minimum(scale, x))


def quantSigm(x, scale):
    return np.maximum(np.minimum(0.5 * (x + scale), scale), 0)


def nonlin(code, x, scale):
    if (code == "quantTanh"):
        return quantTanh(x, scale)
    elif (code == "quantSigm"):
        return quantSigm(x, scale)

fpt = int

def predict_quant(points, I):
    preds = []
    pred_lbls = []
    assert points.ndim == 3

    for i in range(points.shape[0]):
        h = np.array(np.zeros((NUM_HIDDEN, 1)), dtype=fpt)
        # print(h)
        for t in range(points.shape[1]):
            # x = np.array((I * (np.array(points[i][slice(t * stride, t * stride + window)]) - fpt(mean))) / fpt(std),
            #              dtype=fpt).reshape((-1, 1))
            x = np.array((I * (points[i, t] - mean.astype(fpt))) / std.astype(fpt), dtype=fpt).reshape((-1, 1))
            pre = np.array(
                (np.matmul(np.transpose(qW2), np.matmul(np.transpose(qW1), x)) + np.matmul(np.transpose(qU2),
                                                                                           np.matmul(np.transpose(qU1),
                                                                                                     h))) / (
                        q * 1), dtype=fpt)
            h_ = np.array(nonlin(UPDATE_NL, pre + qB_h * I, q * I) / (q), dtype=fpt)
            z = np.array(nonlin(GATE_NL, pre + qB_g * I, q * I) / (q), dtype=fpt)
            h = np.array((np.multiply(z, h) + np.array(np.multiply(fpt(I * zeta) * (I - z) + fpt(I * nu) * I, h_) / I,
                                                       dtype=fpt)) / I, dtype=fpt)

        preds.append(np.matmul(np.transpose(h), qFC_Weight) + qFC_Bias)
        pred_lbls.append(np.argmax(np.matmul(np.transpose(h), qFC_Weight) + qFC_Bias))
    return np.array(preds), np.array(pred_lbls)

# Run test
scale = pow(10, I)
preds, pred_lbls = predict_quant(test_data, scale)
print('Accuracy: ', float((pred_lbls==test_lbls).sum())/test_lbls.shape[0])
confmatrix = getConfusionMatrix(pred_lbls, test_lbls, num_classes)
printFormattedConfusionMatrix(confmatrix)
np.savetxt('C++/out_py.csv', preds.reshape(-1, 2), fmt='%i', delimiter=',')

