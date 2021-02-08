import keras
from keras import backend as K
"""
This code is adapted from the original code in the GitHub repository
https://github.com/CyberZHG/keras-self-attention
"""


class SeqSelfAttention(keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,                 
                 return_attention=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attention_activation=None,
                 **kwargs):
        """Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param return_attention: Whether to return the attention weights for visualization.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self._backend = keras.backend.backend()

        
        self.Wx, self.Wt, self.bh = None, None, None
        self.Wa, self.ba = None, None
       

    def get_config(self):
        config = {
            'units': self.units,
            'return_attention': self.return_attention,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
       
        self._build_attention(input_shape)       
        super(SeqSelfAttention, self).build(input_shape)

    def _build_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)


    def call(self, inputs, mask=None, **kwargs):
        
        alpha = self._emission(inputs)

        if self.attention_activation is not None:
            alpha = self.attention_activation(alpha)
        
        # Equation 2 in the paper
        # \alpha_{r, r'} = = \text{softmax}(\alpha_{r, r'})
        alpha = K.exp(alpha - K.max(alpha, axis=-1, keepdims=True))
        a = alpha / K.sum(alpha, axis=-1, keepdims=True)

        # Equation 2 in the paper
        # \mathbf{c}_r = \sum_{r'} \alpha_{r, r'} \bar{f}_{r'}
        c_r = K.batch_dot(a, inputs)
        

        if self.return_attention:
            return [c_r, a]
        return c_r

    def _emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # Equation 2 in the paper
        # \beta_{r, r'} = \tanh(\bar{f}_r^T W_\bata + \bar{f}_{r'}^T W_{\beta'} + b_\beta)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)        
        beta = K.tanh(q + k + self.bh)
        
        # Equation 2 in paper
        # the computation inside Softmax
        # \alpha_{r, r'} = W_\alpha \beta_{r, r'} + b_\alpha
        alpha = K.reshape(K.dot(beta, self.Wa) + self.ba, (batch_size, input_len, input_len))
        
        return alpha

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}
