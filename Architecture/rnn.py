import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell
import utils as utils


class FastGRNNCell(RNNCell):
    '''
    FastGRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    gate_non_linearity = nonlinearity for the gate can be chosen from
    [tanh, sigmoid, relu, quantTanh, quantSigm]
    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)
    zetaInit = init for zeta, the scale param
    nuInit = init for nu, the translation param

    FastGRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    z_t = gate_nl(Wx_t + Uh_{t-1} + B_g)
    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = z_t*h_{t-1} + (sigmoid(zeta)(1-z_t) + sigmoid(nu))*h_t^

    W and U can further parameterised into low rank version by
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2)
    '''

    def __init__(self, hidden_size, gate_non_linearity="sigmoid",
                 update_non_linearity="tanh", wRank=None, uRank=None,
                 zetaInit=1.0, nuInit=-4.0, name="FastGRNN"):
        super(FastGRNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._gate_non_linearity = gate_non_linearity
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._zetaInit = zetaInit
        self._nuInit = nuInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def gate_non_linearity(self):
        return self._gate_non_linearity

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastGRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastGRNNcell"):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)
            # Init zeta to 6.0 and nu to -6.0 if this doesn't give good
            # results. The inits are hyper-params.
            zeta_init = init_ops.constant_initializer(
                self._zetaInit, dtype=tf.float32)
            self.zeta = vs.get_variable("zeta", [1, 1], initializer=zeta_init)

            nu_init = init_ops.constant_initializer(
                self._nuInit, dtype=tf.float32)
            self.nu = vs.get_variable("nu", [1, 1], initializer=nu_init)

            pre_comp = wComp + uComp

            bias_gate_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_gate = vs.get_variable(
                "B_g", [1, self._hidden_size], initializer=bias_gate_init)
            z = utils.gen_non_linearity(pre_comp + self.bias_gate,
                                  self._gate_non_linearity)

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = utils.gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)

            new_h = z * state + (math_ops.sigmoid(self.zeta) * (1.0 - z) +
                                 math_ops.sigmoid(self.nu)) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_gate, self.bias_update])
        Vars.extend([self.zeta, self.nu])

        return Vars


class FastRNNCell(RNNCell):
    '''
    FastRNN Cell with Both Full Rank and Low Rank Formulations
    Has multiple activation functions for the gates
    hidden_size = # hidden units

    update_non_linearity = nonlinearity for final rnn update
    can be chosen from [tanh, sigmoid, relu, quantTanh, quantSigm]

    wRank = rank of W matrix (creates two matrices if not None)
    uRank = rank of U matrix (creates two matrices if not None)
    alphaInit = init for alpha, the update scalar
    betaInit = init for beta, the weight for previous state

    FastRNN architecture and compression techniques are found in
    FastGRNN(LINK) paper

    Basic architecture is like:

    h_t^ = update_nl(Wx_t + Uh_{t-1} + B_h)
    h_t = sigmoid(beta)*h_{t-1} + sigmoid(alpha)*h_t^

    W and U can further parameterised into low rank version by 
    W = matmul(W_1, W_2) and U = matmul(U_1, U_2) 
    '''

    def __init__(self, hidden_size, update_non_linearity="tanh",
                 wRank=None, uRank=None, alphaInit=-3.0, betaInit=3.0,
                 name="FastRNN"):
        super(FastRNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._update_non_linearity = update_non_linearity
        self._num_weight_matrices = [1, 1]
        self._wRank = wRank
        self._uRank = uRank
        self._alphaInit = alphaInit
        self._betaInit = betaInit
        if wRank is not None:
            self._num_weight_matrices[0] += 1
        if uRank is not None:
            self._num_weight_matrices[1] += 1
        self._name = name

    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def update_non_linearity(self):
        return self._update_non_linearity

    @property
    def wRank(self):
        return self._wRank

    @property
    def uRank(self):
        return self._uRank

    @property
    def num_weight_matrices(self):
        return self._num_weight_matrices

    @property
    def name(self):
        return self._name

    @property
    def cellType(self):
        return "FastRNN"

    def call(self, inputs, state):
        with vs.variable_scope(self._name + "/FastRNNcell"):

            if self._wRank is None:
                W_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W = vs.get_variable(
                    "W", [inputs.get_shape()[-1], self._hidden_size],
                    initializer=W_matrix_init)
                wComp = math_ops.matmul(inputs, self.W)
            else:
                W_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W1 = vs.get_variable(
                    "W1", [inputs.get_shape()[-1], self._wRank],
                    initializer=W_matrix_1_init)
                W_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.W2 = vs.get_variable(
                    "W2", [self._wRank, self._hidden_size],
                    initializer=W_matrix_2_init)
                wComp = math_ops.matmul(
                    math_ops.matmul(inputs, self.W1), self.W2)

            if self._uRank is None:
                U_matrix_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U = vs.get_variable(
                    "U", [self._hidden_size, self._hidden_size],
                    initializer=U_matrix_init)
                uComp = math_ops.matmul(state, self.U)
            else:
                U_matrix_1_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U1 = vs.get_variable(
                    "U1", [self._hidden_size, self._uRank],
                    initializer=U_matrix_1_init)
                U_matrix_2_init = init_ops.random_normal_initializer(
                    mean=0.0, stddev=0.1, dtype=tf.float32)
                self.U2 = vs.get_variable(
                    "U2", [self._uRank, self._hidden_size],
                    initializer=U_matrix_2_init)
                uComp = math_ops.matmul(
                    math_ops.matmul(state, self.U1), self.U2)

            alpha_init = init_ops.constant_initializer(
                self._alphaInit, dtype=tf.float32)
            self.alpha = vs.get_variable(
                "alpha", [1, 1], initializer=alpha_init)

            beta_init = init_ops.constant_initializer(
                self._betaInit, dtype=tf.float32)
            self.beta = vs.get_variable("beta", [1, 1], initializer=beta_init)

            pre_comp = wComp + uComp

            bias_update_init = init_ops.constant_initializer(
                1.0, dtype=tf.float32)
            self.bias_update = vs.get_variable(
                "B_h", [1, self._hidden_size], initializer=bias_update_init)
            c = utils.gen_non_linearity(
                pre_comp + self.bias_update, self._update_non_linearity)

            new_h = math_ops.sigmoid(self.beta) * \
                state + math_ops.sigmoid(self.alpha) * c
        return new_h, new_h

    def getVars(self):
        Vars = []
        if self._num_weight_matrices[0] == 1:
            Vars.append(self.W)
        else:
            Vars.extend([self.W1, self.W2])

        if self._num_weight_matrices[1] == 1:
            Vars.append(self.U)
        else:
            Vars.extend([self.U1, self.U2])

        Vars.extend([self.bias_update])
        Vars.extend([self.alpha, self.beta])

        return Vars

