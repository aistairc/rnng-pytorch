#!/usr/bin/env python3
import numpy as np
import torch
import pickle
from collections import defaultdict
import json
from utils import pad_items, clean_number, berkeley_unk_conv
from action_dict import TopDownActionDict

class Vocabulary(object):
  """
  This vocabulary prohibits registering a new token during lookup.
  Vocabulary should be constructed from a set of tokens with counts (w2c), a dictionary
  from a word to its count in the training data. (or anything)
  """
  def __init__(self, w2c_list, pad = '<pad>', unkmethod = 'unk', unktoken = '<unk>',
               specials = []):
    self.pad = pad
    self.padding_idx = 0
    self.specials = specials
    self.unkmethod = unkmethod
    self.unktoken = unktoken
    if self.unkmethod == 'unk':
      if unktoken not in specials:
        specials.append(unktoken)

    assert isinstance(w2c_list, list)
    self.i2w = [self.pad] + specials + [w for w, _ in w2c_list]
    self.w2i = dict([(w, i) for i, w in enumerate(self.i2w)])
    self.w2c = dict(w2c_list)
    self.i2c = dict([(self.w2i[w], c) for w, c in self.w2c.items()])

  def to_unk(self, w):
    if self.unkmethod == 'unk':
      return self.unktoken
    else:
      return berkeley_unk_conv(w)

  def to_unk_id(self, w_id):
    if self.unkmethod == 'unk':
      return self.unk_id
    else:
      if 1 <= w_id < 1+len(self.specials):
        return w_id
      else:
        return self.get_id(self.i2w[w_id])

  def size(self):
    return len(self.i2w)

  def get_id(self, w):
    if w not in self.w2i:
      w = self.to_unk(w)
    return self.w2i[w]

  def get_count_from_id(self, w_id):
    if w_id not in self.i2c:
      return 0
    else:
      return self.i2c[w_id]

  def get_count(self, w):
    if w not in self.w2c:
      return 0
    else:
      return self.w2c[w]

  # for serialization
  def list_w2c(self):
    return [(w, self.get_count(w)) for w in self.i2w[1+len(self.specials):]]

  def dump(self, fn):
    with open(fn, 'wt') as o:
      o.write(self.pad + '\n')
      o.write(self.unkmethod + '\n')
      o.write(self.unktoken + '\n')
      o.write(' '.join(self.specials) + '\n')
      for w, c in self.list_w2c():
        o.write('{}\t{}\n'.format(w, c))

  def to_json_dict(self):
    return {'pad': self.pad,
            'unkmethod': self.unkmethod,
            'unktoken': self.unktoken,
            'specials': self.specials,
            'word_count': self.list_w2c()}

  @staticmethod
  def load(self, fn):
    with open(fn) as f:
      lines = [line for line in f]
    pad, unkmethod, unktoken, specials = [l.strip() for l in line[:4]]
    specials = [w for w in specials]
    def parse_line(line):
      w, c = line[:-1].split()
      return w, int(c)
    w2c_list = [parse_line(line) for line in lines[4:]]
    return Vocabulary(w2c_list, pad, unkmethod, unktoken, specials)

  @staticmethod
  def from_data_json(data):
    d = data['vocab']
    return Vocabulary(d['word_count'], d['pad'], d['unkmethod'], d['unktoken'],
                      d['specials'])

class Sentence(object):
  def __init__(self, orig_tokens, tokens, token_ids, tags, actions=[], action_ids=[], tree_str=None):
    self.orig_tokens = orig_tokens
    self.tokens = tokens
    self.token_ids = token_ids
    self.tags = tags
    self.actions = actions
    self.action_ids = action_ids
    self.tree_str = tree_str  # original annotation

  @staticmethod
  def from_json(j):
    return Sentence(j['orig_tokens'],
                    j['tokens'],
                    j['token_ids'],
                    j.get('tags', []),
                    j.get('actions', []),
                    j.get('action_ids', []),
                    j.get('tree_str', None))

  def random_unked(self, vocab):
    def unkify_rand(w_id):
      c = vocab. get_count_from_id(w_id)
      if c == 0 or (np.random.rand() < 1 / (1 + c)):
        return vocab.to_unk_id(w_id)
      else:
        return w_id
    return [unkify_rand(i) for i in self.token_ids]

  def to_dict(self):
    return {'orig_tokens': self.orig_tokens, 'tokens': self.tokens,
            'token_ids': self.token_ids,'tags': self.tags,
            'actions': self.actions, 'action_ids': self.action_ids,
            'tree_str': self.tree_str}

class Dataset(object):
  def __init__(self, sents, batch_size, vocab, action_dict, random_unk=False, prepro_args={}):
    self.sents = sents
    self.batch_size = batch_size
    self.vocab = vocab
    self.action_dict = action_dict
    self.random_unk = random_unk
    self.prepro_args = prepro_args  # keeps which process is performed.

    self.vocab_size = vocab.size()
    self.len_to_idxs = self._get_len_to_idxs()
    self.num_batches = self.get_num_batches()

  @staticmethod
  def from_json(data_file, batch_size, vocab=None, action_dict=None, random_unk=False):
    """If vocab and action_dict are provided, they are not loaded from data_file.
    This is for sharing these across train/valid/test sents.

    If random_unk = True, replace a token in a sentence to unk with a probability
    inverse proportional to the frequency in the training data.
    TODO: add custom unkifier?
    """
    j = json.load(open(data_file))
    sents = [Sentence.from_json(s) for s in j['sentences']]
    vocab = vocab or Vocabulary.from_data_json(j)
    action_dict = action_dict or TopDownActionDict(j['nonterminals'])

    return Dataset(sents, batch_size, vocab, action_dict, random_unk, j['args'])

  @staticmethod
  def from_text_file(text_file, batch_size, vocab, action_dict, tagger_fn=None,
                     prepro_args = {}):
    """tagger_fn is a function receiving a sentence and returning POS tags.
    If Not provided, dummy tags (X) are provided.
    """
    tagger_fn = tagger_fn or (lambda tokens: ['X' for _ in tokens])
    sents = []
    with open(text_file) as f:
      for line in f:
        orig_tokens = line.strip().split()
        tokens = orig_tokens[:]
        if prepro_args.get('lowercase', False):
          tokens = [t.lower() for t in tokens]
        if prepro_args.get('replace_num', False):
          tokens = [clean_number(t) for t in tokens]
        token_ids = [vocab.get_id(t) for t in tokens]
        tags = tagger_fn(tokens)
        sent = Sentence(orig_tokens, tokens, token_ids, tags)
        sents.append(sent)
    return Dataset(sents, batch_size, vocab, action_dict, False, prepro_args)

  def get_num_batches(self):
    b = 0
    for _, idxs in self.len_to_idxs.items():
      if len(idxs) % self.batch_size == 0:
        b += len(idxs) // self.batch_size
      else:
        b += (len(idxs) // self.batch_size) + 1
    return b

  def batches(self, shuffle=True):
    yield from self.batches_helper(self.len_to_idxs, shuffle)

  def test_batches(self, block_size = 1000):
    assert block_size > 0
    """
    Sents are first segmented (chunked) by block_size, and then, mini-batched.
    Since each batch contains batch_idx, we can recover the original order of
    data, by processing output grouping this size.

    This may be useful when outputing the parse results (approximately) streamly,
    by dumping to stdout (or file) at once for every 100~1000 sentences.
    Below is an such example to dump parse results by keeping the original sentence
    order.
    ```
    batch_size = 3
    block_size = 1000
    parses = []
    idxs = []
    for token, action, idx in dataset.test_batches(block_size):
      parses.extend(parse(token))
      idxs.extend(idx)
      if len(idxs) >= block_size:
        assert len(idxs) <= block_size
        parse_idx_to_sent_idx = sorted(list(enmearte(idxs)), key=lambda x:x[1])
        orig_order_parses = [parses[sent_idx] for (parse_idx, sent_idx) in parse_idx_to_sent_idx]
        # process orig_order_parses (e.g., print)
        parses = []
        idxs = []
    ```
    """
    for offset in range(0, len(self.sents), block_size):
      end = min(len(self.sents), offset + block_size)
      len_to_idxs = self._get_len_to_idxs(range(offset, end))
      yield from self.batches_helper(len_to_idxs, False, True)

  def batches_helper(self, len_to_idxs, shuffle=True, test=False):
    # `len_to_idxs` summarizes sentence length to idx in `self.sents`.
    # This may be a subset of sentences, or full sentences.
    batches = []
    for length, idxs in len_to_idxs.items():
      if shuffle:
        idxs = np.random.permutation(idxs)
      for begin in range(0, len(idxs), self.batch_size):
        batches.append(idxs[begin:begin+self.batch_size])
    if shuffle:
        batches = np.random.permutation(batches)

    if self.random_unk:
      def conv_sent(i):
        return self.sents[i].random_unked(self.vocab)
    else:
      def conv_sent(i):
        return self.sents[i].token_ids

    for batch_idx in batches:
      token_ids = [conv_sent(i) for i in batch_idx]
      ret = (torch.tensor(token_ids, dtype=torch.long),)
      if not test:
        action_ids = [self.sents[i].action_ids for i in batch_idx]
        ret += (torch.tensor(self._pad_action_ids(action_ids), dtype=torch.long),)
      ret += (batch_idx,)
      yield ret

  def _get_len_to_idxs(self, sent_idxs = []):
    if len(sent_idxs) == 0:
      sent_idxs = range(len(self.sents))
    len_to_idxs = defaultdict(list)
    for i in sent_idxs:
      sent = self.sents[i]
      len_to_idxs[len(sent.token_ids)].append(i)
    return len_to_idxs

  def _pad_action_ids(self, action_ids):
    action_ids, _ = pad_items(action_ids, self.action_dict.padding_idx)
    return action_ids

  def __len__(self):
    return self.num_batches
