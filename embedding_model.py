from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def raw_data(data_path=None):

  train_path = os.path.join("train.txt")
  test_path = os.path.join("test.txt")
  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = word_to_id
  return train_data, test_data, vocabulary, len(vocabulary) 

def batch_gen(raw_data, batch_size):
    i = 0 
    while i < (len(raw_data) - 1 ):
        
        x = raw_data[i: i+batch_size]
        y = raw_data[i+1:i+batch_size+1]

        if (len(x) > len(y)):
            yield x[:-1], y
            return

        yield x, y 
        i += batch_size 



class EmbeddingModel(object):

    def __init__(self, sess, vocab):

        self.sess = sess
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.define_graph()


    def define_graph(self): 
        self.inpt = tf.placeholder(tf.int32, [None])
        self.out = tf.placeholder(tf.int32, [None])


        self.embedding = tf.truncated_normal([self.vocab_size, 30], stddev=0.1)
        self.embd = tf.nn.embedding_lookup(self.embedding, self.inpt) 


        self.W1 = tf.Variable(tf.truncated_normal([30, 100], stddev=0.1))
        self.B1 = tf.Variable(tf.constant(.1, shape=[100]))

        self.H1 = tf.nn.relu(tf.matmul(self.embd, self.W1) + self.B1)


        self.W2 = tf.Variable(tf.truncated_normal([100, self.vocab_size], stddev=0.1))
        self.B2 = tf.Variable(tf.constant(.1, shape=[self.vocab_size]))


        self.logits = tf.matmul(self.H1, self.W2) + self.B2 

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.out) 
        
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

        self.perplexity = tf.exp(self.cost)

    def train_step(self, sess, feed_dict):

        cost, _ = sess.run([self.cost, self.train_op], feed_dict=feed_dict)
        return cost 




def main(): 

    sess = tf.InteractiveSession()

    train_data, test_data, vocab, vocab_len = raw_data()
    model = EmbeddingModel(sess, vocab)

    sess.run(tf.initialize_all_variables())

    epochs = 1
    count = 0
    for i in range(epochs):
        for x, y in batch_gen(train_data, 20):
            cost = model.train_step(sess, feed_dict={model.inpt:x, model.out:y})
            count += 1
            if (count % 100) == 0:
                print("Count: %d, Train Perplexity: %d" % (count, np.exp(cost)))

            if (count % 1000) == 0:
                for foo, bar in batch_gen(test_data, len(test_data)):
                    perplexity = sess.run(model.perplexity, 
                        feed_dict={model.inpt:test_data[:-1], model.out:test_data[1:]})
                    print("Test Perplexity %d" % perplexity)

main()

