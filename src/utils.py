import logging
import random
from io import open

import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5):
    """Penalizes deviations from margin for each logit.
    Each wrong logit costs its distance to margin. For negative logits margin is
    0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
    margin is 0.4 from each side.
    Args:
    labels: tensor, one hot encoding of ground truth.
    raw_logits: tensor, model.py predictions in range [0, 1]
    margin: scalar, the margin after subtracting 0.5 from raw_logits.
    downweight: scalar, the factor for negative cost.
    Returns:
    A tensor with cost for each data point of shape [batch_size].
    """
    logits = raw_logits - 0.5
    positive_cost = labels * tf.cast(tf.less(logits, margin),
                                     tf.float32) * tf.pow(logits - margin, 2)
    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def siamese_loss(labels, cosine_similarity, margin=0.25):
    """
    Penalize deviations from the expected label for each cosine similarity result
    positve label 1 == cosine similarity 1
    negative label 0 == cosine simialrity -1
    :param labels: tensors, true label 0/1 for each instance
    :param cosine_similarity: expected cosine simialrity, ranging from -1 to 1
    :param margin: scalar
    :return: A tensor with cost for each data point of shape [batch_size]
    """
    return labels * (0.25 * tf.square(1 - cosine_similarity)) + (1 - labels) * tf.square(
        tf.maximum(cosine_similarity - margin, 0))


def evaluateTopN(pred_scores, true_scores, chunks, k_num):
    pred_scores_topk = []
    true_scores_topk = []
    scores_all = []

    for c in chunks.keys():
        true_index = [index for index, value in enumerate(list(true_scores[chunks[c]])) if value == 1]
        filtered_true_index = list(filter(lambda x: x < k_num, true_index))
        true_scores_topk.append(len(filtered_true_index))

        sorted_scores = sorted(pred_scores[chunks[c]])[::-1]
        pred_index = [sorted_scores.index(pred_scores[chunks[c]][index]) for index in true_index]
        filtered_pred_index = list(filter(lambda x: x < k_num, pred_index))
        pred_scores_topk.append(len(filtered_pred_index))

        scores_all.append(len(true_index))

    topkp = np.array(pred_scores_topk) * 1.0 / k_num
    topkr = np.array(pred_scores_topk) * 1.0 / np.array(scores_all)
    topkf1 = np.mean(np.nan_to_num((2 * topkp * topkr) / (topkp + topkr)))
    topkp = np.mean(topkp)
    topkr = np.mean(topkr)
    logging.info("\tP@" + str(k_num) + ":" + str(topkp) + "\tR@" + str(k_num) + ":" + str(topkr) + "\tF@" + str(
        k_num) + ":" + str(topkf1))
    return topkp, topkr, topkf1


def createVocabulary(input_path, output_path, pad=True, unk=True):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'rb') as fd, \
            open(output_path, 'w+') as out:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(fd, binary=True)
        vocab = model.index2entity
        init_vocab = []
        if pad:
            init_vocab.append('_PAD')
        if unk:
            init_vocab.append('_UNK')
        vocab = vocab + init_vocab
        for v in vocab:
            out.write(v + '\n')


def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x, y) for (y, x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}


def sentenceToIds(data, vocab, unk, keyword=None):
    '''
    this version does not deal with context chunking, the contexts are chunked to certain size during dataset preprocessing
    :param data:
    :param vocab:
    :param unk:
    :param context_size:
    :param keyword:
    :return:
    '''
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')
    try:
        keyword_index_new = words.index(keyword)
    except ValueError:
        keyword_index_new = None
    ids = []
    if unk:
        for w in words:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w, vocab['_UNK']))
    else:
        for w in words:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w))

    return ids, keyword_index_new


def padSentence(s, max_length, vocab):
    return s + [vocab['vocab']['_PAD']] * (max_length - len(s))


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


class DataProcessor(object):
    def __init__(self, in_path, in_vocab, shuffle=False):
        self._fd_in = open(in_path, 'r').readlines()
        if shuffle:
            self.shuffle()
        self._in_vocab = in_vocab
        self.end = 0

    def close(self):
        pass

    def shuffle(self):
        random.shuffle(self._fd_in)
        logging.info("In Memory Shuffle")

    def get_chunk(self, a_entity, a, context_size=19):
        indexes = list(map(lambda x: x.split(" ").index(a_entity), a))
        words_list = list()
        for context_idx, context in enumerate(a):
            words = context.split(" ")
            keyword_index = indexes[context_idx]
            words = list(zip(*filter(lambda i: np.abs(i[0] - keyword_index) < context_size, enumerate(words))))[1]
            words_list.append(" ".join(words))
        return words_list

    def get_batch_siamese(self, arg):
        batch_size = arg.batch_size
        batch_a_ids = []
        batch_a_context_ids = []
        batch_a_keyword_index = []
        batch_a_max_len = [0]
        batch_a_len = []

        batch_p_ids = []
        batch_p_context_ids = []
        batch_p_keyword_index = []
        batch_p_max_len = [0]
        batch_p_len = []

        batch_y = []

        for i in range(batch_size):
            try:
                inp = self._fd_in.pop()
            except IndexError:
                self.end = 1
                break

            a, p, a_contexts, p_contexts, y = inp.strip().split("\t")
            a_context = self.get_chunk(a, a_contexts.split("@@")[:arg.doc_size], context_size=arg.context_length)
            p_context = self.get_chunk(p, p_contexts.split("@@")[:arg.doc_size], context_size=arg.context_length)

            a_id, _ = sentenceToIds(a, self._in_vocab, unk=False)
            p_id, _ = sentenceToIds(p, self._in_vocab, unk=False)
            a_context_ids, a_keyword_index = zip(*list(
                map(lambda x: sentenceToIds(x, self._in_vocab, unk=True, keyword=a), a_context)))
            p_context_ids, p_keyword_index = zip(*list(
                map(lambda x: sentenceToIds(x, self._in_vocab, unk=True, keyword=p), p_context)))
            a_max_len = max(map(len, a_context_ids))
            batch_a_ids.append(a_id[0])
            batch_a_context_ids.append(a_context_ids)
            batch_a_keyword_index.append(a_keyword_index)
            batch_a_max_len.append(a_max_len)
            batch_a_len.append(list(map(len, a_context_ids)))

            p_max_len = max(map(len, p_context_ids))
            batch_p_ids.append(p_id[0])
            batch_p_context_ids.append(p_context_ids)
            batch_p_keyword_index.append(p_keyword_index)
            batch_p_max_len.append(p_max_len)
            batch_p_len.append(list(map(len, p_context_ids)))

            batch_y.append(float(y))

        global_a_max_len = max(batch_a_max_len)
        global_p_max_len = max(batch_p_max_len)

        batch_a_context_ids = np.array(list(
            map(lambda xx: list(map(lambda x: padSentence(x, global_a_max_len, self._in_vocab), xx)),
                batch_a_context_ids)))
        batch_p_context_ids = np.array(list(
            map(lambda xx: list(map(lambda x: padSentence(x, global_p_max_len, self._in_vocab), xx)),
                batch_p_context_ids)))
        batch_a_keyword_index = np.array(batch_a_keyword_index)
        batch_p_keyword_index = np.array(batch_p_keyword_index)
        return batch_a_ids, batch_a_context_ids, batch_a_keyword_index, batch_a_len, batch_p_ids, batch_p_context_ids, batch_p_keyword_index, batch_p_len, batch_y

    def get_batch_triple(self, arg):
        batch_size = arg.batch_size
        batch_a_ids = []
        batch_a_context_ids = []
        batch_a_keyword_index = []
        batch_a_max_len = [0]
        batch_a_len = []

        batch_p_ids = []
        batch_p_context_ids = []
        batch_p_keyword_index = []
        batch_p_max_len = [0]
        batch_p_len = []

        batch_n_ids = []
        batch_n_context_ids = []
        batch_n_keyword_index = []
        batch_n_max_len = [0]
        batch_n_len = []

        for i in range(batch_size):
            try:
                inp = self._fd_in.pop()
            except IndexError:
                self.end = 1
                break

            a, p, n, a_contexts, p_contexts, n_contexts = inp.strip().split("\t")
            a_context = self.get_chunk(a, a_contexts.split("@@")[:arg.doc_size], context_size=arg.context_length)
            p_context = self.get_chunk(p, p_contexts.split("@@")[:arg.doc_size], context_size=arg.context_length)
            n_context = self.get_chunk(n, n_contexts.split("@@")[:arg.doc_size], context_size=arg.context_length)

            a_id, _ = sentenceToIds(a, self._in_vocab, unk=False)
            n_id, _ = sentenceToIds(n, self._in_vocab, unk=False)
            p_id, _ = sentenceToIds(p, self._in_vocab, unk=False)
            a_context_ids, a_keyword_index = zip(*list(
                map(lambda x: sentenceToIds(x, self._in_vocab, unk=True, keyword=a), a_context)))
            n_context_ids, n_keyword_index = zip(*list(
                map(lambda x: sentenceToIds(x, self._in_vocab, unk=True, keyword=n), n_context)))
            p_context_ids, p_keyword_index = zip(*list(
                map(lambda x: sentenceToIds(x, self._in_vocab, unk=True, keyword=p), p_context)))
            a_max_len = max(map(len, a_context_ids))
            batch_a_ids.append(a_id[0])
            batch_a_context_ids.append(a_context_ids)
            batch_a_keyword_index.append(a_keyword_index)
            batch_a_max_len.append(a_max_len)
            batch_a_len.append(list(map(len, a_context_ids)))

            n_max_len = max(map(len, n_context_ids))
            batch_n_ids.append(n_id[0])
            batch_n_context_ids.append(n_context_ids)
            batch_n_keyword_index.append(n_keyword_index)
            batch_n_max_len.append(n_max_len)
            batch_n_len.append(list(map(len, n_context_ids)))

            p_max_len = max(map(len, p_context_ids))
            batch_p_ids.append(p_id[0])
            batch_p_context_ids.append(p_context_ids)
            batch_p_keyword_index.append(p_keyword_index)
            batch_p_max_len.append(p_max_len)
            batch_p_len.append(list(map(len, p_context_ids)))

        global_a_max_len = max(batch_a_max_len)
        global_n_max_len = max(batch_n_max_len)
        global_p_max_len = max(batch_p_max_len)

        batch_a_context_ids = np.array(list(
            map(lambda xx: list(map(lambda x: padSentence(x, global_a_max_len, self._in_vocab), xx)),
                batch_a_context_ids)))
        batch_n_context_ids = np.array(list(
            map(lambda xx: list(map(lambda x: padSentence(x, global_n_max_len, self._in_vocab), xx)),
                batch_n_context_ids)))
        batch_p_context_ids = np.array(list(
            map(lambda xx: list(map(lambda x: padSentence(x, global_p_max_len, self._in_vocab), xx)),
                batch_p_context_ids)))
        batch_a_keyword_index = np.array(batch_a_keyword_index)
        batch_p_keyword_index = np.array(batch_p_keyword_index)
        batch_n_keyword_index = np.array(batch_n_keyword_index)
        return batch_a_ids, batch_a_context_ids, batch_a_keyword_index, batch_a_len, batch_p_ids, batch_p_context_ids, batch_p_keyword_index, batch_p_len, batch_n_ids, batch_n_context_ids, \
               batch_n_keyword_index, batch_n_len


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Network
    parser.add_argument("--layer_size", type=int, default=256, help="Network size.", dest='layer_size')
    parser.add_argument("--embed_dim", type=int, default=200, help="Embedding dim.", dest='embed_dim')
    parser.add_argument("--embed_path", type=str, default='skipgram-vec200-mincount5-win5.bin',
                        help="Path to the pretraiend bedding file.", dest='embed_path')
    parser.add_argument("--num_rnn", type=int, default=1, help="Num of layers for stacked RNNs.")
    parser.add_argument("--doc_size", type=int, default=20, help="Num of context for each keyword, up to 20 for wiki")
    parser.add_argument("--context_length", type=int, default=80, help="Length of left/right context")

    # Training Environment
    parser.add_argument("--leaky", type=bool, default=True, help="Leaky unit for softmax")
    parser.add_argument("--optimizer", type=str, default='rmsprop', help="Optimizer.")
    parser.add_argument("--batch_size", type=int, default=26, help="Batch size.")
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

    # Data
    parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
    parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
    parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
    parser.add_argument("--input_file", type=str, default='siamese_contexts.txt', help="Input file name.")

    arg = parser.parse_args()
