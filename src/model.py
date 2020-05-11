# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors


class SiameseModel():
    def __init__(self, arg):
        self.num_hidden_units = arg.layer_size
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.embedding_dim = arg.embed_dim
        self.word_vectors = KeyedVectors.load_word2vec_format("../input_data/" + arg.dataset + "/" + arg.embed_path,
                                                              binary=True, encoding='utf')
        self.word_vectors_matrix = np.concatenate(
            [self.word_vectors.vectors, np.random.uniform(0, 1, [2, self.embedding_dim])], axis=0)
        print("embedding loaded")
        self.embedding = tf.get_variable(name='embedding', shape=self.word_vectors_matrix.shape,
                                         initializer=tf.constant_initializer(self.word_vectors_matrix),
                                         trainable=True)
        self.is_leaky = arg.leaky
        self.doc_size = arg.doc_size
        self.batch_size = None
        self.encoder_type = arg.model_type
        self.input_a = tf.placeholder(tf.int32, [self.batch_size, ], 'input_a')
        self.input_n = tf.placeholder(tf.int32, [self.batch_size, ], 'input_n')
        self.input_a_context = tf.placeholder(tf.int32, [self.batch_size, self.doc_size, None],
                                              'input_a_context')  # second dim is the number of sentences, third dim is the word id [2,5,16]
        self.input_n_context = tf.placeholder(tf.int32, [self.batch_size, self.doc_size, None],
                                              'input_n_context')  # second dim is the number of sentences, third dim is the word id [2,5,16]
        self.input_a_len = tf.placeholder(tf.int32, [self.batch_size, self.doc_size],
                                          'input_a_len')  # second dim the the number of sentences [2,5]
        self.input_n_len = tf.placeholder(tf.int32, [self.batch_size, self.doc_size],
                                          'input_n_len')  # second dim the the number of sentences [2,5]
        self.input_a_keyword_index = tf.placeholder(tf.int32, [self.batch_size, self.doc_size], 'input_a_keyword_index')
        self.input_n_keyword_index = tf.placeholder(tf.int32, [self.batch_size, self.doc_size], 'input_n_keyword_index')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, ], 'input_y')

    def _softmax_with_mask(self, logits, lens, axis=-1):
        # sequence_len is always on the last dimension
        # logits [16, 74, 18]; lens [16,74]
        exp_logits = tf.exp(logits)  # [16, 74, 18]
        mask = tf.sequence_mask(lens, maxlen=tf.shape(logits)[axis], dtype=tf.float32)  # [16, 74, 18]
        masked_exp_logits = tf.multiply(exp_logits, mask)  # [16, 74, 18]
        masked_exp_logits_sum = tf.reduce_sum(masked_exp_logits, axis)  # [16, 74]
        return tf.clip_by_value(tf.div(masked_exp_logits, tf.expand_dims(masked_exp_logits_sum, axis)), 1e-37,
                                1e+37)  # [16, 18, 74]

    def _leaky_routing(self, logits, output_dim):
        """Adds extra dimmension to routing logits.
        This enables contexts to be routed to the extra dim if they are not a
        good fit for any of the contexts for the other entity.
        """
        # leak is a zero matrix with same shape as logits except dim(2) = 1 because
        # of the reduce_sum.
        leak = tf.zeros_like(logits, optimize=True)  # (4,5)
        leak = tf.reduce_sum(leak, axis=1, keep_dims=True)  # （4，1）
        leaky_logits = tf.concat([leak, logits], axis=1)  # （4，6）
        leaky_routing = tf.nn.softmax(leaky_logits, dim=1)  # （4，6）
        return tf.split(leaky_routing, [1, output_dim], 1)[1]  # （4，5）

    def _doc_embedding(self, doc, doc_len, keyword_index, num_units, batch_size, doc_size):
        with tf.variable_scope("doc_embedding", reuse=tf.AUTO_REUSE):
            doc = tf.reshape(doc, [batch_size * doc_size, -1])  # (4*5,?)
            doc_len = tf.reshape(doc_len, [batch_size * doc_size, ])  # (4*5,)
            keyword_index = tf.reshape(keyword_index, [batch_size * doc_size, 1])  # (4*5,1)
            embed_doc = tf.nn.embedding_lookup(self.embedding, doc, max_norm=1)  # [10, ?, 200]
            self.fw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            self.bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            doc_outputs, doc_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                      self.bw_cell, inputs=embed_doc,
                                                                      sequence_length=doc_len, dtype=tf.float32)
            if self.encoder_type == 'confluence':
                range_index = tf.reshape(tf.range(start=0, limit=batch_size * doc_size, dtype=tf.int32), [-1, 1])
                gather_index = tf.concat([range_index, keyword_index], axis=1)
                fw_outputs = tf.gather_nd(doc_outputs[0], gather_index)
                bw_outputs = tf.gather_nd(doc_outputs[1], gather_index)
                doc_outputs_at_keyword_index = tf.reshape(tf.concat([fw_outputs, bw_outputs], axis=1),
                                                          [batch_size, doc_size, -1])
            elif self.encoder_type in ['bilstm']:
                doc_outputs_at_keyword_index = tf.reshape(tf.concat([doc_states[0].h, doc_states[1].h], axis=1),
                                                          [batch_size, doc_size, -1])
            return doc_outputs_at_keyword_index  # [4, 5, 200]

    def build_model(self):
        a = self._doc_embedding(self.input_a_context, self.input_a_len, self.input_a_keyword_index,
                                self.num_hidden_units, tf.shape(self.input_a_context)[0], self.doc_size)
        n = self._doc_embedding(self.input_n_context, self.input_n_len, self.input_n_keyword_index,
                                self.num_hidden_units, tf.shape(self.input_a_context)[0], self.doc_size)
        an = tf.matmul(a, tf.transpose(n, [0, 2, 1]))  # (4, 5, 200) * (4,200,5') = (4, 5, 5') suppose h2 has 5' docs
        a_pool = tf.reduce_max(an, axis=2, keep_dims=False)  # (4,5)
        n_pool = tf.reduce_max(an, axis=1, keep_dims=False)  # (4,5')
        if self.is_leaky:
            a_prob = self._leaky_routing(a_pool, self.doc_size)
            n_prob = self._leaky_routing(n_pool, self.doc_size)
        else:
            a_prob = tf.nn.softmax(a_pool)
            n_prob = tf.nn.softmax(n_pool)
        a_weighted = tf.multiply(a, tf.expand_dims(a_prob, 2))
        n_weighted = tf.multiply(n, tf.expand_dims(n_prob, 2))
        c_a = tf.reduce_sum(a_weighted, axis=1)
        c_n = tf.reduce_sum(n_weighted, axis=1)
        normalize_c_a = tf.nn.l2_normalize(c_a, 1)
        normalize_c_n = tf.nn.l2_normalize(c_n, 1)
        cos_similarity = tf.reduce_sum(tf.multiply(normalize_c_a, normalize_c_n), axis=1)
        return a, cos_similarity


class TripletModel():
    def __init__(self, arg):
        self.num_hidden_units = arg.layer_size
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        self.embedding_dim = arg.embed_dim
        self.word_vectors = KeyedVectors.load_word2vec_format("../input_data/" + arg.dataset + "/" + arg.embed_path,
                                                              binary=True, encoding='utf')
        self.word_vectors_matrix = np.concatenate(
            [self.word_vectors.vectors, np.random.uniform(0, 1, [2, self.embedding_dim])], axis=0)
        print("embedding loaded")
        self.embedding = tf.get_variable(name='embedding', shape=self.word_vectors_matrix.shape,
                                         initializer=tf.constant_initializer(self.word_vectors_matrix),
                                         trainable=True)
        self.is_leaky = arg.leaky
        self.doc_size = arg.doc_size
        self.batch_size = None
        self.encoder_type = arg.model_type

        self.input_a = tf.placeholder(tf.int32, [self.batch_size, ], 'input_a')
        self.input_a_context = tf.placeholder(tf.int32, [self.batch_size, self.doc_size, None],
                                              'input_a_context')  # second dim is the number of sentences, third dim is the word id e.g. [2,5,16]
        self.input_a_keyword_index = tf.placeholder(tf.int32, [self.batch_size, self.doc_size], 'input_a_keyword_index')
        self.input_a_len = tf.placeholder(tf.int32, [self.batch_size, self.doc_size],
                                          'input_a_len')  # second dim the the number of sentences e.g. [2,5]

        self.input_n = tf.placeholder(tf.int32, [self.batch_size, ], 'input_n')
        self.input_n_context = tf.placeholder(tf.int32, [self.batch_size, self.doc_size, None],
                                              'input_n_context')  # second dim is the number of sentences, third dim is the word id e.g. [2,5,16]
        self.input_n_keyword_index = tf.placeholder(tf.int32, [self.batch_size, self.doc_size], 'input_n_keyword_index')
        self.input_n_len = tf.placeholder(tf.int32, [self.batch_size, self.doc_size],
                                          'input_n_len')  # second dim the the number of sentences e.g. [2,5]

        self.input_p = tf.placeholder(tf.int32, [self.batch_size, ], 'input_p')
        self.input_p_context = tf.placeholder(tf.int32, [self.batch_size, self.doc_size, None],
                                              'input_p_context')  # second dim is the number of sentences, third dim is the word id e.g.[2,5,16]
        self.input_p_keyword_index = tf.placeholder(tf.int32, [self.batch_size, self.doc_size], 'input_p_keyword_index')
        self.input_p_len = tf.placeholder(tf.int32, [self.batch_size, self.doc_size],
                                          'input_p_len')  # second dim the the number of sentences e.g.[2,5]

    def _softmax_with_mask(self, logits, lens, axis=-1):
        # sequence_len is always on the last dimension, batch_size: 16; doc_size: 74; max_len: 18
        # logits [16, 74, 18]; lens [16,74]
        exp_logits = tf.exp(logits)  # [16, 74, 18]
        mask = tf.sequence_mask(lens, maxlen=tf.shape(logits)[axis], dtype=tf.float32)  # [16, 74, 18]
        masked_exp_logits = tf.multiply(exp_logits, mask)  # [16, 74, 18]
        masked_exp_logits_sum = tf.reduce_sum(masked_exp_logits, axis)  # [16, 74]
        return tf.clip_by_value(tf.div(masked_exp_logits, tf.expand_dims(masked_exp_logits_sum, axis)), 1e-37,
                                1e+37)  # [16, 18, 74]

    def _leaky_routing(self, logits, output_dim):
        # leak is a zero matrix with same shape as logits except dim(2) = 1 because
        # of the reduce_sum.
        leak = tf.zeros_like(logits, optimize=True)
        leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
        leaky_logits = tf.concat([leak, logits], axis=1)
        leaky_routing = tf.nn.softmax(leaky_logits, dim=1)
        return tf.split(leaky_routing, [1, output_dim], 1)[1]

    def _doc_embedding(self, doc, doc_len, keyword_index, num_units, batch_size, doc_size):
        with tf.variable_scope("doc_embedding", reuse=tf.AUTO_REUSE):
            doc = tf.reshape(doc, [batch_size * doc_size, -1])  # (4*5,?)
            doc_len = tf.reshape(doc_len, [batch_size * doc_size, ])  # (4*5,)
            keyword_index = tf.reshape(keyword_index, [batch_size * doc_size, 1])  # (4*5,1)
            embed_doc = tf.nn.embedding_lookup(self.embedding, doc)  # [10, ?, 200]
            self.fw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            self.bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
            doc_outputs, doc_states = tf.nn.bidirectional_dynamic_rnn(self.fw_cell,
                                                                      self.bw_cell, inputs=embed_doc,
                                                                      sequence_length=doc_len, dtype=tf.float32)
            if self.encoder_type == 'confluence':
                range_index = tf.reshape(tf.range(start=0, limit=batch_size * doc_size, dtype=tf.int32), [-1, 1])
                gather_index = tf.concat([range_index, keyword_index], axis=1)
                fw_outputs = tf.gather_nd(doc_outputs[0], gather_index)
                bw_outputs = tf.gather_nd(doc_outputs[1], gather_index)
                doc_outputs_at_keyword_index = tf.reshape(tf.concat([fw_outputs, bw_outputs], axis=1),
                                                          [batch_size, doc_size, -1])
            elif self.encoder_type in ['bilstm']:
                doc_outputs_at_keyword_index = tf.reshape(tf.concat([doc_states[0].h, doc_states[1].h], axis=1),
                                                          [batch_size, doc_size, -1])
            return doc_outputs_at_keyword_index  # [4, 5, 200]

    def _matching_network(self, m1, m2):
        with tf.variable_scope('matching_network', reuse=tf.AUTO_REUSE):
            m12 = tf.matmul(m1,
                            tf.transpose(m2, [0, 2, 1]))  # (4, 5, 200) * (4,200,5') = (4, 5, 5') suppose h2 has 5' docs
            m1_pool = tf.reduce_max(m12, axis=2, keep_dims=False)  # (4,5)
            m2_pool = tf.reduce_max(m12, axis=1, keep_dims=False)  # (4,5')
            if self.is_leaky:
                m1_prob = self._leaky_routing(m1_pool, self.doc_size)
                m2_prob = self._leaky_routing(m2_pool, self.doc_size)
            else:
                m1_prob = tf.nn.softmax(m1_pool)
                m2_prob = tf.nn.softmax(m2_pool)
            m1_weighted = tf.multiply(m1, tf.expand_dims(m1_prob, 2))
            m2_weighted = tf.multiply(m2, tf.expand_dims(m2_prob, 2))
            c_m1 = tf.reduce_sum(m1_weighted, axis=1)
            c_m2 = tf.reduce_sum(m2_weighted, axis=1)
            normalize_c_m1 = tf.nn.l2_normalize(c_m1, 1)
            normalize_c_m2 = tf.nn.l2_normalize(c_m2, 1)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_c_m1, normalize_c_m2), axis=1)
            return cos_similarity

    def build_model(self):
        a = self._doc_embedding(self.input_a_context, self.input_a_len, self.input_a_keyword_index,
                                self.num_hidden_units, tf.shape(self.input_a_context)[0], self.doc_size)
        n = self._doc_embedding(self.input_n_context, self.input_n_len, self.input_n_keyword_index,
                                self.num_hidden_units, tf.shape(self.input_a_context)[0], self.doc_size)
        p = self._doc_embedding(self.input_p_context, self.input_p_len, self.input_p_keyword_index,
                                self.num_hidden_units, tf.shape(self.input_a_context)[0], self.doc_size)
        cos_an = self._matching_network(a, n)
        cos_ap = self._matching_network(a, p)
        return cos_an, cos_ap


if __name__ == '__main__':
    tf.reset_default_graph()

    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Network
    parser.add_argument("--layer_size", type=int, default=256, help="Network size.", dest='layer_size')
    parser.add_argument("--embed_dim", type=int, default=200, help="Embedding dim.", dest='embed_dim')
    parser.add_argument("--embed_path", type=str, default='../input_data/wiki/wiki-skipgram-vec200-mincount5-win5.bin',
                        help="Path to the pretraiend bedding file.", dest='embed_path')
    parser.add_argument("--num_rnn", type=int, default=1, help="Num of layers for stacked RNNs.")
    parser.add_argument("--doc_size", type=int, default=5, help="Num of context for each keyword, up to 20 for wiki")

    # Training Environment
    parser.add_argument("--leaky", type=bool, default=False, help="Leaky unit for softmax")
    parser.add_argument("--optimizer", type=str, default='rmsprop', help="Optimizer.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max epochs to train.")
    parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                        help="Disable early stop, which is based on sentence level accuracy.")
    parser.add_argument("--patience", type=int, default=10, help="Patience to wait before stop.")
    parser.add_argument("--run_name", type=str, default='temp_run', help="Run name.")

    # Model and Vocab
    parser.add_argument("--dataset", type=str, default='wiki', help="Note, if you don't want to use this part, "
                                                                    "enter --dataset=''. It can not be None""")
    parser.add_argument("--model_path", type=str, default='../model', help="Path to save model.")
    parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

    # Data
    parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
    parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
    parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
    parser.add_argument("--input_file", type=str, default='triple_contexts.txt', help="Input file name.")

    arg = parser.parse_args()
    model = TripletModel(arg)
    model.build_model()
