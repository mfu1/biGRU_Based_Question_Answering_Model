from keras import backend as K
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, InputLayer, LSTM, Embedding, Dropout, Activation, Masking, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
sys.path.append("/Users/meif/Desktop/SI 630 NLP/Project/Code/layers/")

from SharedWeight import SharedWeight
from VariationalDropout import VariationalDropout
from QuestionAttnGRU import QuestionAttnGRU
from SelfAttnGRU import SelfAttnGRU
from QuestionPooling import QuestionPooling


class RNet(Model):
    def __init__(self, inputs=None, outputs=None,
                       N=None, M=None, C=None, 
                       word2vec_dim=None, label_size=None, embedding_matrix=None,
                       hdim=None, dropout_rate=None, output_type=None,
                       unroll=False, **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(RNet, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       **kwargs)
            return
        
        '''Dimensions'''
        B = None
        H = hdim
        W = word2vec_dim

        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')
        VQ_r = SharedWeight(size=(H, H), name='VQ_r')
        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v, VQ_r]

        
        P_vecs = Input(shape=(N,), name='P_vecs')
        P = Embedding(len(embedding_matrix), W, 
                      weights=[embedding_matrix], trainable=False,
                      input_length=N) (P_vecs)

        Q_vecs = Input(shape=(M,), name='Q_vecs')
        Q = Embedding(len(embedding_matrix), W, 
                      weights=[embedding_matrix], trainable=False,
                      input_length=M) (Q_vecs)

        input_placeholders = [P_vecs, Q_vecs]

        uP = Masking() (P)
        for i in range(1):
            uP = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate, unroll=False)) (uP)
        uP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uP') (uP)

        uQ = Masking() (Q)
        for i in range(1):
            uQ = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=dropout_rate, unroll=False)) (uQ)
        uQ = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uQ') (uQ)

        vP = QuestionAttnGRU(units=H,
                             return_sequences=True,
                             unroll=unroll) ([uP, uQ,
                                              WQ_u, WP_v, WP_u, v, W_g1])
        vP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, H), name='vP') (vP)

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True,
                                       unroll=unroll)) ([vP, vP,
                                                         WP_v, WPP_v, v, W_g2])

        hP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='hP') (hP)

#         rQ = QuestionPooling() ([uQ, WQ_u, WQ_v, v, VQ_r])
#         rQ = Dropout(rate=dropout_rate, name='rQ') (rQ)
        
        if output_type == "bi":
            gP = Bidirectional(GRU(units=H,
                                  return_sequences=True,
                                  unroll=unroll)) (hP)
            preds = TimeDistributed(Dense(1, activation='sigmoid')) (gP)
        elif output_type == "multi":
            gP = Bidirectional(GRU(units=H,
                                  return_sequences=False,
                                  unroll=unroll)) (hP)
            preds = Dense(label_size, activation='softmax')(gP)
        
        inputs = input_placeholders + shared_weights
        outputs = preds

        super(RNet, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

