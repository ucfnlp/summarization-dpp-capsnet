import tensorflow as tf
from keras import backend as K
from keras.engine import Model
from keras.layers import Input, Dense, Conv1D, Embedding, Layer
from keras.layers import Concatenate, Flatten, Lambda, Multiply, Dropout
from keras.layers import SpatialDropout1D, Activation, LSTM, TimeDistributed
from keras.optimizers import adam
from keras.utils import plot_model

from capsulelayers import squash # CapsuleLayer, Length,  PrimaryCap, Length, Mask

def max_axis1(mat):    
    return K.max(mat, axis=1, keepdims=True)

def abs_diff(vects):
    x, y = vects
    return K.abs(x-y)

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),                                     
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b += K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class CapsNetTextSim():
    def __init__(self, logger, p, h, 
                 save_folder, folder_name,
                 word_embedding_weights,                 
                 filters=[3,4,5,6,7], n_filter_out=30,
                 capsule_num=5, capsule_dim=6, routings=3,                 
                 lr=None, dr_rate=0.2, 
                 voca_size=50003, lstm_layer_num=1, lstm_hidden_unit=256, draw_summary_network=True):
        """
        A Capsule Network for Text Similarity.
        :param logger: logger
        :param p: Text sentence length
        :param h: Hypothesis sentence length
        :param save_folder: directory to save model summary
        :param folder_name: directory name to save model summary
        :param word_embedding_weights: weights vectors (300 in length)
        :param filters: 1D convolution kernels
        :param n_filter_out: 1D convolution output size
        :param capsule_num: number of output capsule
        :param capsule_dim: number of output capsule dimension
        :param routings: number of routing iterations
        :param lr: initial learning rate
        :param dr_rate: dropout rate
        :param voca_size: vocabulary size
        :param lstm_layer_num: number of LSTM layer
        :param lstm_hidden_unit: number of LSTM hidden units
        :param draw_summary_network: output summary network        
        """
        self.logger = logger
        self.p = p
        self.h = h
        self.save_folder = save_folder
        self.folder_name = folder_name
        self.word_embedding_weights = word_embedding_weights
        self.filters = filters
        self.n_filter_out = n_filter_out        
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.lr = lr
        self.dr_rate = dr_rate                
        self.voca_size = voca_size
        self.lstm_layer_num = lstm_layer_num
        self.lstm_hidden_unit = lstm_hidden_unit
        self.draw_summary_network = draw_summary_network
        
        self.loss = []
        self.metrics = {}
        self.optimizer = adam #adam rmsprop
                
    
    def __call__(self):
        self.build_network()
        
        if self.draw_summary_network:
            self.model.summary()
    
    def genInputs(self):
        self.premise_word_input    = Input(shape=(self.p,), dtype='int32', name='PremiseWordInput')
        self.hypothesis_word_input = Input(shape=(self.h,), dtype='int32', name='HypothesisWordInput')
        
    def embedding(self, input):        
        word_embedding = Embedding(input_dim=self.word_embedding_weights.shape[0],
                                       output_dim=self.word_embedding_weights.shape[1],
                                       weights=[self.word_embedding_weights],
                                       trainable=False)(input)        
        word_embedding = SpatialDropout1D(self.dr_rate)(word_embedding)        
        return word_embedding
        
    def conv1D_stack_decoder(self):
        input = Input(shape=(self.p,), dtype='int32')
        word_embedding = self.embedding(input)
        
        # 1D Conv stack
        convs = []
        for filter in self.filters:            
            # 1D conv
            conv = Conv1D(filters=self.n_filter_out, kernel_size=filter, strides=1, padding='same', #activation='relu',
                          name='1Dconv_'+str(filter))(word_embedding)            
            convs.append(conv)
        concated = Concatenate(name='concated')(convs)
        
        # decoder
        max_ = Lambda(max_axis1, name='max_over_time')(concated)
        flat = Flatten()(max_)
        layer_h0 = Dense(self.lstm_hidden_unit, activation='relu')(flat)
        layer_c0 = Dense(self.lstm_hidden_unit, activation='relu')(flat)
        initial_state = [layer_h0, layer_c0]
                
        decoder_lstm = LSTM(self.lstm_hidden_unit, return_sequences=True, return_state=True)
        for i in range(self.lstm_layer_num):
            if i==0:
                x = word_embedding
            x, h_state, c_state = decoder_lstm(x, initial_state=initial_state)
            initial_state = [h_state, c_state]        
        
        x = TimeDistributed(Dense(self.voca_size))(x)		# (None, 42, 50k)
        decoder_out = Activation('softmax', name='decoder_out')(x)
        
        model = Model(inputs=input, outputs=[concated, decoder_out], name='conv1d_stack_decoder')
        plot_model(model, to_file='{}/{}_conv1d_stack_decoder.pdf'.format(self.save_folder, self.folder_name), show_shapes=True)
        return model       
        
    def build_network(self):
        self.genInputs()        
        self.premise_mask = Input(shape=(self.p,), name='PremiseMask')
        self.hypothesis_mask = Input(shape=(self.h,), name='HypothesisMask')
        self.premise_exact_input    = Input(shape=(self.p-2,), name='PremiseExactInput')
        self.hypothesis_exact_input = Input(shape=(self.h-2,),  name='HypothesisExactInput')
        p_exact_expanded = Lambda(lambda x: K.expand_dims(x, 1), name='p_exact_expand')(self.premise_exact_input)
        h_exact_expanded = Lambda(lambda x: K.expand_dims(x, 1), name='h_exact_expand')(self.hypothesis_exact_input)
                
        # conv1D stack
        conv1D_stack_decoder_layer = self.conv1D_stack_decoder()
        conca_p, dec_out_p = conv1D_stack_decoder_layer(self.premise_word_input)
        conca_h, dec_out_h = conv1D_stack_decoder_layer(self.hypothesis_word_input)
        
        max_p = Lambda(max_axis1, name='max_p')(conca_p)    # sentence level
        max_h = Lambda(max_axis1, name='max_h')(conca_h)    # sentence level
        
        mul_ph = Multiply(name='multiply_ph')([max_p, max_h])
        diff_ph = Lambda(abs_diff, name='abs_diff_ph')([max_p, max_h])
                
        conca_ph = Concatenate(name='concat_ph', axis=2)([conca_p, conca_h])        # 1: [88, 500] 2: [44, 1000]
        squashed = Lambda(squash, name='squash')(conca_ph)
        
        # capsule network
        caps_out = Capsule(num_capsule=self.capsule_num, dim_capsule=self.capsule_dim*2, 
                          routings=self.routings, share_weights=True, name='capsule_layer_1way')(squashed)
        
        dr = Dropout(self.dr_rate)(caps_out)
        max_cap = Lambda(max_axis1, name='max_capsout')(dr)
                
        # concat_all, binary prediction
        conca_all = Concatenate(name='concat_all')([max_cap, max_p, max_h, mul_ph, diff_ph, p_exact_expanded, h_exact_expanded])
        
        flat = Flatten()(conca_all)
        dense100 = Dense(100, activation='relu', name='dense100')(flat)        
        pred = Dense(1, activation='sigmoid', name='bin_pred')(dense100)
        
        # model output
        self.model = Model(inputs=[self.premise_word_input, self.hypothesis_word_input,
                                   self.premise_exact_input, self.hypothesis_exact_input], 
                           outputs=[pred, dec_out_p, dec_out_h])
        self.pred_model = Model(inputs=[self.premise_word_input, self.hypothesis_word_input, 
                                        self.premise_exact_input, self.hypothesis_exact_input], 
                           outputs=[pred])
        self.loss = ['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
        self.metrics = ['acc'] #{'capsnet_1':'acc'}
                        
        # compile the model        
        self.model.compile(optimizer=self.optimizer(lr=self.lr),
                           loss=self.loss, metrics=self.metrics, loss_weights=[1., 0.00005, 0.00005])