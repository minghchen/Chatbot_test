# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading disc_data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w.lower() for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from disc_data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: disc_data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each disc_data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_quary_path, target_answer_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
  """Tokenize disc_data file and turn into token-ids using given vocabulary file.

  This function loads disc_data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the disc_data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_quary_path) or not gfile.Exists(target_answer_path) :
    print("Tokenizing data in %s" % data_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_quary_path, mode="w") as tokens_quary_file:
        with gfile.GFile(target_answer_path, mode="w") as tokens_answer_file:
          counter = 0
          last_line = ' '
          for line in data_file:
            counter += 1
            if counter % 10000 == 0:
              print("  tokenizing line %d" % counter)
              print("  quary: ", last_line, "  answer: ", line)
            last_line = line
            token_ids = sentence_to_token_ids(line, vocabulary, tokenizer,
                                              normalize_digits)
            if(counter%2==1): tokens_quary_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
            else: tokens_answer_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def read_data(config, tokenized_quary_path, tokenized_answer_path, max_size=None):
  """Read data from tokenized file and put into buckets.

  Args:
    source_path: path to the files with token-ids.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in config.buckets]

  with gfile.GFile(tokenized_quary_path, mode="r") as fq:
    with gfile.GFile(tokenized_answer_path, mode="r") as fa:
      source, target = fq.readline(), fa.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        for bucket_id, (source_size, target_size) in enumerate(config.buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = fq.readline(), fa.readline()
  return data_set

def prepare_data(gen_config):
  """Get dialog data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the English vocabulary to create and use.

  Returns:
    A tuple of 3 elements:
      (1) path to the token-ids for chat training data-set,
      (2) path to the token-ids for chat development data-set,
      (3) path to the chat vocabulary file
  """
  # Get dialog data to the specified directory.
  data_dir = gen_config.train_dir
  vocabulary_size = gen_config.vocab_size
  train_path = os.path.join(data_dir, "chat")
  dev_path = os.path.join(data_dir, "chat_test")

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab%d.in" % vocabulary_size)
  create_vocabulary(vocab_path, train_path + ".in", vocabulary_size)
  # loading the vocabulary into memory
  vocab, rev_vocab = initialize_vocabulary(vocab_path)

  # Create token ids for the training data.
  train_quary_ids_path = train_path + ("_quary.ids%d.in" % vocabulary_size)
  train_answer_ids_path = train_path + ("_answer.ids%d.in" % vocabulary_size)
  if not gfile.Exists(train_quary_ids_path) or not gfile.Exists(train_answer_ids_path):
    data_to_token_ids(train_path + ".in", train_quary_ids_path, train_answer_ids_path, vocab)

  # Create token ids for the development data.
  dev_quary_ids_path = dev_path + ("_quary.ids%d.in" % vocabulary_size)
  dev_answer_ids_path = dev_path + ("_answer.ids%d.in" % vocabulary_size)
  if not gfile.Exists(dev_quary_ids_path) or not gfile.Exists(dev_answer_ids_path):
    data_to_token_ids(dev_path + ".in", dev_quary_ids_path, dev_answer_ids_path, vocab)

  # Read disc_data into buckets and compute their sizes.
  print("Reading development and training gen_data")
  dev_set = read_data(gen_config, dev_quary_ids_path, dev_answer_ids_path)
  train_set = read_data(gen_config, train_quary_ids_path, train_answer_ids_path)

  return vocab, rev_vocab, dev_set, train_set

def read_disc_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [PAD_ID] * (query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [PAD_ID] * (answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [PAD_ID] * (answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set

def prepare_disc_data_path(data_dir):
    query_train_ids_path = os.path.join(data_dir, "train.query")
    answer_train_ids_path = os.path.join(data_dir, "train.answer")
    gen_train_ids_path = os.path.join(data_dir, "train.gen")

    return query_train_ids_path, answer_train_ids_path, gen_train_ids_path

def prepare_disc_data(config):
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    # use gen train vocab
    vocab_path = os.path.join(config.gen_train_dir, "vocab%d.in" % config.vocab_size)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path = prepare_disc_data_path(config.train_dir)
    query_set, answer_set, gen_set = read_disc_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set, vocab, rev_vocab


import random
from six.moves import xrange
import numpy as np
def get_batch(config, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of time-major vectors, while
    data here contains single batch-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(config.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = config.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    inputs_len, target_len = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(config.batch_size):
        encoder_input, decoder_input = random.choice(data[bucket_id])

        # Encoder inputs are padded and then reversed.
        encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(encoder_input + encoder_pad)
        inputs_len.append(len(encoder_input))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)
        target_len.append(len(decoder_input)+1)

    # Now we create time-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(config.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(config.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(config.batch_size, dtype=np.float32)
        for batch_idx in xrange(config.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, inputs_len, target_len

def clean(inputs, ID):
    resps = []
    seq_tokens_t = []
    for col in range(len(inputs[0])):
        seq_tokens_t.append([inputs[row][col] for row in range(len(inputs))])

    for seq in seq_tokens_t:
        if ID in seq:
            resps.append(seq[:seq.index(ID)])
        else:
            resps.append(seq)
    return resps
