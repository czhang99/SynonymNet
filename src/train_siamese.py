# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import *
from utils import evaluateTopN
from utils import DataProcessor
from src.model import SiameseModel
from src.utils import createVocabulary
from src.utils import loadVocabulary

np.seterr(divide='ignore', invalid='ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
logging.info('\ngpu = ' + str(gpu_id))

parser = argparse.ArgumentParser(allow_abbrev=False)

# Network
parser.add_argument("--layer_size", type=int, default=256, help="Network size.", dest='layer_size')
parser.add_argument("--embed_dim", type=int, default=200, help="Embedding dim.", dest='embed_dim')
parser.add_argument("--embed_path", type=str, default='skipgram-vec200-mincount5-win5.bin',
                    help="Path to the pretraiend bedding file.", dest='embed_path')
parser.add_argument("--num_rnn", type=int, default=1, help="Num of layers for stacked RNNs.")
parser.add_argument("--doc_size", type=int, default=5, help="Num of context for each keyword, up to 20 for wiki")
parser.add_argument("--context_length", type=int, default=80, help="Length of left/right context")

# Training Environment
parser.add_argument("--leaky", type=bool, default=True, help="Leaky unit for softmax")
parser.add_argument("--model_type", type=str, default='confluence', help="""confluence(default) | bilstm | gru
                                                                    confluence: confluence encoder is used
                                                                    bilstm: bilstm is used
                                                                    gru: gru is used""")

parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=200, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                    help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=10, help="Patience to wait before stop.")
parser.add_argument("--run_name", type=str, default='temp_run', help="Run name.")

# Model and Vocab
parser.add_argument("--dataset", type=str, default='wiki', help="Note, if you don't want to use this part, "
                                                                "enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='../model', help="Path to save model.")

# Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='siamese_contexts.txt', help="Input file name.")
parser.add_argument("--inference_file", type=str, default='knn-siamese_contexts.txt', help="Inference file name.")
parser.add_argument("--inference_label_file", type=str, default='knn-siamese_contexts.txt', help="Inference file name.")
parser.add_argument("--k_num", type=int, default=5, help="Number of knn during inference")

arg = parser.parse_args()

logs_path = '../log/train_siamese/' + arg.run_name
if os.path.isdir(logs_path):
    sys.exit(1)

# Print arguments
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

if arg.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(learning_rate=arg.learning_rate)
else:
    print('unknown optimizer!')
    exit(1)

if arg.model_type in ['confluence', 'bilstm', 'gru']:
    pass
else:
    print('unknown model type!')
    exit(1)

# full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'wiki':
    print('use wiki+freebase dataset')
elif arg.dataset == 'pubmed':
    print('use pubmed+umls dataset')
else:
    print('use own dataset: ', arg.dataset)

full_train_path = os.path.join('../input_data', arg.dataset, arg.train_data_path, arg.input_file)
full_valid_path = os.path.join('../input_data', arg.dataset, arg.valid_data_path, arg.input_file)
full_test_path = os.path.join('../input_data', arg.dataset, arg.test_data_path, arg.input_file)
full_inference_path = os.path.join('../input_data', arg.dataset, arg.test_data_path, arg.inference_file)
full_inference_label_path = os.path.join('../input_data', arg.dataset, arg.test_data_path, arg.inference_label_file)
createVocabulary("../input_data/" + arg.dataset + "/" + arg.embed_path, "../input_data/" + arg.dataset + "/in_vocab",
                 pad=True, unk=True)
in_vocab = loadVocabulary("../input_data/" + arg.dataset + "/in_vocab")
logging.info("vocab created")

# Create Training Model
with tf.variable_scope('siamese_model'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    model = SiameseModel(arg)
    outputs = model.build_model()

with tf.variable_scope('loss'):
    cossim = outputs[-1]
    m = tf.constant(1.0, tf.float32)
    loss = model.input_y * 0.25 * tf.square(1 - cossim) + (1 - model.input_y) * (
                tf.subtract(tf.constant(1.0, tf.float32), tf.cast(tf.greater(cossim - m, 0.0), tf.float32)) * tf.square(
            cossim) + tf.cast(tf.greater(cossim - m, 0.0), tf.float32) * 0.0)

params = tf.trainable_variables()
gradients = tf.gradients(loss, params)
clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
gradient_norm = norm
update = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
training_outputs = [global_step, outputs, loss, update, gradient_norm]
inference_outputs = [cossim, loss]
saver = tf.train.Saver()

# Start Training
with tf.Session(config=tf.ConfigProto(allow_soft_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')

    epochs = 0
    eval_loss = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    valid_err = 1
    test_err = 1

    while True:
        if data_processor == None:
            data_processor = DataProcessor(full_train_path, in_vocab, shuffle=True)
        a_ids_data, a_context_ids_data, a_keyword_index, a_len_data, n_ids_data, n_context_ids_data, n_keyword_index, n_len_data, y = data_processor.get_batch_siamese(
            arg)
        if len(a_ids_data) != 0:
            feed_dict = {model.input_a.name: a_ids_data,
                         model.input_a_context.name: a_context_ids_data,
                         model.input_a_keyword_index.name: a_keyword_index,
                         model.input_a_len.name: a_len_data,
                         model.input_n.name: n_ids_data,
                         model.input_n_context.name: n_context_ids_data,
                         model.input_n_keyword_index.name: n_keyword_index,
                         model.input_n_len.name: n_len_data,
                         model.input_y.name: y
                         }
            ret = sess.run(training_outputs, feed_dict)
            eval_loss += np.mean(ret[2])
            line += arg.batch_size
            step = ret[0]
            num_loss += 1

        # end of an epoch
        if data_processor.end == 1:
            line = 0
            data_processor.shuffle("../input_data/" + arg.dataset + "/" + arg.train_data_path + "/" + arg.input_file)
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Training Loss: ' + str(eval_loss / num_loss))
            num_loss = 0
            eval_loss = 0.0

            def valid(full_path, in_vocab):
                data_processor_valid = DataProcessor(full_path, in_vocab)
                pred_scores = []
                true_scores = []
                eval_loss = 0
                num_loss = 0

                while True:
                    a_ids_data, a_context_ids_data, a_keyword_index, a_len_data, p_ids_data, p_context_ids_data, \
                    p_keyword_index, p_len_data, y_data = \
                        data_processor_valid.get_batch_siamese(arg)
                    if len(a_ids_data) != 0:
                        feed_dict = {model.input_a.name: a_ids_data,
                                     model.input_a_context.name: a_context_ids_data,
                                     model.input_a_keyword_index.name: a_keyword_index,
                                     model.input_a_len.name: a_len_data,
                                     model.input_n.name: p_ids_data,
                                     model.input_n_context.name: p_context_ids_data,
                                     model.input_n_keyword_index.name: p_keyword_index,
                                     model.input_n_len.name: p_len_data,
                                     model.input_y: y_data
                                     }
                        ret = sess.run(inference_outputs, feed_dict)
                        eval_loss += np.mean(ret[1])
                        num_loss += 1
                        pred_scores.append(ret[0])
                        true_scores.append(y_data)

                    if data_processor_valid.end == 1:
                        break

                pred_scores = np.concatenate(pred_scores)
                true_scores = np.concatenate(true_scores)
                import sklearn
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_scores, pred_scores, pos_label=1)
                auc = sklearn.metrics.auc(fpr, tpr)
                map = average_precision_score(true_scores, pred_scores, average='micro')
                df = pd.DataFrame({'model': 'siamese', 'score': pred_scores, 'class': true_scores})
                logging.info('Loss: ' + str(eval_loss / num_loss))
                logging.info('AUC: ' + str(auc))
                logging.info('MAP: ' + str(map))

                data_processor_valid.close()
                return (eval_loss / num_loss), auc, df


            def inference(full_path, full_inference_label_file, in_vocab):
                data_processor_valid = DataProcessor(full_path, in_vocab)
                pred_scores = []

                while True:
                    a_ids_data, a_context_ids_data, a_keyword_index, a_len_data, p_ids_data, p_context_ids_data, \
                    p_keyword_index, p_len_data, y_data = \
                        data_processor_valid.get_batch_siamese(arg)
                    if len(a_ids_data) != 0:
                        feed_dict = {model.input_a.name: a_ids_data,
                                     model.input_a_context.name: a_context_ids_data,
                                     model.input_a_keyword_index.name: a_keyword_index,
                                     model.input_a_len.name: a_len_data,
                                     model.input_n.name: p_ids_data,
                                     model.input_n_context.name: p_context_ids_data,
                                     model.input_n_keyword_index.name: p_keyword_index,
                                     model.input_n_len.name: p_len_data,
                                     model.input_y: y_data
                                     }
                        ret = sess.run(inference_outputs, feed_dict)
                        pred_scores.append(ret[0])

                    if data_processor_valid.end == 1:
                        break

                pred_scores = np.concatenate(pred_scores)
                df = pd.read_csv(full_inference_label_file, sep='\t', header=None)
                true_scores = df.iloc[:, 4]
                chunks = df.groupby(df.iloc[:, 0]).groups
                for k_num in [1, 5, 10, 20]:
                    topkp, topkr, topkf1 = evaluateTopN(pred_scores, true_scores, chunks, k_num)
                data_processor_valid.close()
                return

            logging.info('Validation:')
            epoch_valid_loss, valid_auc, valid_df = valid(full_valid_path, in_vocab)

            if epoch_valid_loss >= valid_err:
                no_improve += 1
                logging.info('No improvement on validation loss')
            else:
                valid_err = epoch_valid_loss
                no_improve = 0
                logging.info('New best validation loss')

            if epochs == arg.max_epochs:
                break

            if arg.early_stop == True:
                if no_improve > arg.patience:
                    break

            logging.info('Test:')
            epoch_test_loss, test_auc, test_df = valid(full_test_path, in_vocab)