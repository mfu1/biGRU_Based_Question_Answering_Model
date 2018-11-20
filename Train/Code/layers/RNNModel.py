from keras import backend as K
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import concatenate, Dense, Input, InputLayer, LSTM, Embedding, Dropout, Activation, Masking, RepeatVector, Reshape
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


class RNNModel(Model):
    def __init__(self, inputs=None, outputs=None,
                       N=None, M=None, C=None, 
                       word2vec_dim=None, label_size=None, embedding_matrix=None,
                       hdim=None, dropout_rate=None, output_type=None,
                       unroll=False, **kwargs):
        # Load model from config
        if inputs is not None and outputs is not None:
            super(RNNModel, self).__init__(inputs=inputs,
                                           outputs=outputs,
                                           **kwargs)
            return

        H = hdim
        W = word2vec_dim
        
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
        uP = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               dropout=dropout_rate, unroll=False)) (uP)
        uP = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uP') (uP)

        uQ = Masking() (Q)
        uQ = Bidirectional(GRU(units=H,
                               return_sequences=True,
                               dropout=dropout_rate, unroll=False)) (uQ)
        uQ = VariationalDropout(rate=dropout_rate, noise_shape=(None, 1, 2 * H), name='uQ') (uQ)
        
        merged = concatenate([uP, uQ], axis=1)
        merged = Bidirectional(GRU(units=H, 
                                   return_sequences=False,
                                   dropout=dropout_rate, unroll=False)) (merged)
        preds = Dense(label_size, activation='softmax')(merged)

        inputs = input_placeholders
        outputs = preds

        super(RNNModel, self).__init__(inputs=inputs, outputs=outputs, **kwargs)
