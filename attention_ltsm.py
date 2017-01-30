from keras import backend as K
from keras.layers import LSTM

class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, **kwargs):
        self.attention_vec = attention_vec
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        assert hasattr(self.attention_vec, '_keras_shape')
        attention_dim = self.attention_vec._keras_shape[1]

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s,
                                   self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = K.tanh(K.dot(h, self.U_a) + attention + self.b_a)
        s = K.exp(K.dot(m, self.U_s) + self.b_s)
        h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants
