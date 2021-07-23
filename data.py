#!/usr/bin/env python3
import sys
import numpy as np
import math
import torch
import pickle
from collections import defaultdict
import json
from utils import pad_items, clean_number, berkeley_unk_conv, berkeley_unk_conv2, get_subword_boundary_mask
from action_dict import TopDownActionDict, InOrderActionDict

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

    if self.unkmethod == 'unk':
      self.unk_id = self.w2i[self.unktoken]

  def id_to_word(self, i):
    return self.i2w[i]

  def to_unk(self, w):
    if self.unkmethod == 'unk':
      return self.unktoken
    elif self.unkmethod == 'berkeleyrule':
      return berkeley_unk_conv(w)
    elif self.unkmethod == 'berkeleyrule2':
      return berkeley_unk_conv2(w)

  def to_unk_id(self, w_id):
    if self.unkmethod == 'unk':
      return self.unk_id
    else:
      if 1 <= w_id < 1+len(self.specials):
        return w_id
      else:
        return self.get_id(berkeley_unk_conv(self.i2w[w_id]))

  def size(self):
    return len(self.i2w)

  def get_id(self, w):
    if w not in self.w2i:
      w = self.to_unk(w)
      if w not in self.w2i:
        # Back-off to a general unk token when converted unk-token is not registered in the
        # vocabulary (which happens when an unseen unk token is generated at test time).
        w = self.unktoken
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

class SentencePieceVocabulary(object):
  def __init__(self, sp_model_path):
    import sentencepiece as spm
    self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    self.padding_idx = self.sp.pad_id()
    self.pad = self.sp.id_to_piece(self.padding_idx)
    self.unkmethod = 'subword'
    self.unk_id = self.sp.unk_id()
    self.unktoken = self.sp.id_to_piece(self.unk_id)

  def id_to_word(self, i):
    return self.sp.id_to_piece(i)

  def to_unk(self, w):
    assert False, "SentencePieceVocabulary should not call to_unk()"

  def to_unk_id(self, w_id):
    assert False, "SentencePieceVocabulary should not call to_unk_id()"

  def size(self):
    return self.sp.get_piece_size()

  def get_id(self, w):
    return self.sp.piece_to_id(w)

  def get_count_from_id(self, w_id):
    return 1

  def get_count(self, w):
    return 1

class Sentence(object):
  def __init__(self, orig_tokens, tokens, token_ids, tags,
               actions=None, action_ids=None, tree_str=None,
               max_stack_size=-1, is_subword_end=None):
    self.orig_tokens = orig_tokens
    self.tokens = tokens
    self.token_ids = token_ids
    self.tags = tags
    self.actions = actions or []
    self.action_ids = action_ids or []
    self.tree_str = tree_str  # original annotation
    self.max_stack_size = max_stack_size

    if is_subword_end is not None:
      assert isinstance(is_subword_end, list)
    else:
      assert self.tokens is not None
      is_subword_end = get_subword_boundary_mask(self.tokens)
    self.is_subword_end = is_subword_end

  @staticmethod
  def from_json(j, oracle='top_down'):
    if oracle == 'top_down':
      actions = j.get('actions', [])
      action_ids = j.get('action_ids', [])
    elif oracle == 'in_order':
      actions = j.get('in_order_actions', [])
      action_ids = j.get('in_order_action_ids', [])
    return Sentence(j['orig_tokens'],
                    j['tokens'],
                    j['token_ids'],
                    j.get('tags', []),
                    actions,
                    action_ids,
                    j.get('tree_str', None),
                    j.get('max_stack_size', -1),
                    j.get('is_subword_end', None))

  def random_unked(self, vocab):
    def unkify_rand(w_id):
      c = vocab.get_count_from_id(w_id)
      if c == 0 or (np.random.rand() < 1 / (1 + c)):
        return vocab.to_unk_id(w_id)
      else:
        return w_id
    return [unkify_rand(i) for i in self.token_ids]

  # def get_word_end_mask(self):
  #   if any('▁' in t for t in self.tokens):
  #     # subword-tokenized
  #     return ['▁' in t for t in self.tokens]
  #   else:
  #     return [True for t in self.tokens]

  def get_subword_span_index(self):
    """
    Given tokens = ["In▁", "an▁", "Oct", ".▁", "19▁"],
    return [[0], [1], [2, 3], [4]].
    """
    idxs = []
    cur_word_idx = []
    for i, is_end in enumerate(self.is_subword_end):
      cur_word_idx.append(i)
      if is_end:
        idxs.append(cur_word_idx)
        cur_word_idx = []
    assert idxs[-1][-1] == len(self.tokens)-1  # tokens is subword-segmented.
    return idxs

  def to_dict(self):
    return {'orig_tokens': self.orig_tokens, 'tokens': self.tokens,
            'token_ids': self.token_ids,'tags': self.tags,
            'actions': self.actions, 'action_ids': self.action_ids,
            'tree_str': self.tree_str, 'max_stack_size': self.max_stack_size}

class Dataset(object):
  def __init__(self, sents, batch_size, vocab, action_dict, random_unk=False, prepro_args={},
               batch_token_size = 15000, batch_action_size = 50000, batch_group='same_length',
               max_length_diff = 20, group_sentence_size = 1024):
    self.sents = sents
    self.batch_size = batch_size
    self.batch_token_size = batch_token_size  # This bounds the batch size by the number of tokens.
    self.batch_action_size = batch_action_size
    self.group_sentence_size = group_sentence_size
    self.vocab = vocab
    self.action_dict = action_dict
    self.random_unk = random_unk
    self.prepro_args = prepro_args  # keeps which process is performed.

    self.use_subwords = isinstance(self.vocab, SentencePieceVocabulary)

    self.vocab_size = vocab.size()
    if batch_group == 'same_length':
      self.len_to_idxs = self._get_len_to_idxs()
    elif batch_group == 'similar_length' or batch_group == 'similar_action_length':
      use_action_len = batch_group == 'similar_action_length'
      self.len_to_idxs = self._get_grouped_len_to_idxs(
        use_action_len=use_action_len, max_length_diff=max_length_diff)
    elif batch_group == 'random':
      self.len_to_idxs = self._get_random_len_to_idxs()
    self.num_batches = self.get_num_batches()

  @staticmethod
  def from_json(data_file, batch_size, vocab=None, action_dict=None, random_unk=False,
                oracle='top_down', batch_group='same_length', batch_token_size=15000,
                batch_action_size = 50000, max_length_diff = 20, group_sentence_size = 1024):
    """If vocab and action_dict are provided, they are not loaded from data_file.
    This is for sharing these across train/valid/test sents.

    If random_unk = True, replace a token in a sentence to unk with a probability
    inverse proportional to the frequency in the training data.
    TODO: add custom unkifier?
    """
    def new_action_dict(nonterminals):
      if oracle == 'top_down':
        return TopDownActionDict(nonterminals)
      elif oracle == 'in_order':
        return InOrderActionDict(nonterminals)

    j = Dataset._load_json_helper(data_file)
    sents = [Sentence.from_json(s, oracle) for s in j['sentences']]
    vocab = vocab or Vocabulary.from_data_json(j)
    action_dict = action_dict or new_action_dict(j['nonterminals'])

    return Dataset(sents, batch_size, vocab, action_dict, random_unk, j['args'],
                   batch_group=batch_group, batch_token_size=batch_token_size,
                   batch_action_size=batch_action_size, max_length_diff=max_length_diff,
                   group_sentence_size=group_sentence_size)

  @staticmethod
  def _load_json_helper(path):
    def read_jsonl(f):
      data = {}
      sents = []
      for line in f:
        o = json.loads(line)
        k = o['key']
        if k == 'sentence':
          o['is_subword_end'] = get_subword_boundary_mask(o['tokens'])
          # Unused values are discarded here (for reducing memory for larger data).
          o['tokens'] = o['orig_tokens'] = o['tree_str'] = o['actions'] = o['in_order_actions'] = o['tags'] = None
          sents.append(o)
        else:
          # key except 'sentence' should only appear once
          assert k not in data
          data[k] = o['value']
      data['sentences'] = sents
      return data

    try:
      with open(path) as f:
        # Old format => a single fat json object containing everything.
        return json.load(f)
    except json.decoder.JSONDecodeError:
      with open(path) as f:
        # New format => jsonl
        return read_jsonl(f)

  @staticmethod
  def from_text_file(text_file, batch_size, vocab, action_dict, tagger_fn = None,
                     prepro_args = {}, batch_token_size = 15000, batch_group = 'same_length'):
    """tagger_fn is a function receiving a sentence and returning POS tags.
    If Not provided, dummy tags (X) are provided.
    """
    tagger_fn = tagger_fn or (lambda tokens: ['X' for _ in tokens])
    sents = []
    with open(text_file) as f:
      for line in f:
        orig_tokens = line.strip().split()
        tokens = orig_tokens[:]
        if isinstance(vocab, SentencePieceVocabulary):
          tokens = vocab.sp.encode(' '.join(tokens), out_type=str)
          token_ids = vocab.sp.piece_to_id(tokens)
        else:
          if prepro_args.get('lowercase', False):
            tokens = [t.lower() for t in tokens]
          if prepro_args.get('replace_num', False):
            tokens = [clean_number(t) for t in tokens]
          token_ids = [vocab.get_id(t) for t in tokens]
        tags = tagger_fn(orig_tokens)
        sent = Sentence(orig_tokens, tokens, token_ids, tags)
        sents.append(sent)
    return Dataset(sents, batch_size, vocab, action_dict, False, prepro_args, batch_token_size,
                   batch_group=batch_group)

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

  def test_batches(self, block_size = 1000, max_length_diff=20):
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
      len_to_idxs = self._get_grouped_len_to_idxs(range(offset, end), max_length_diff=max_length_diff)
      yield from self.batches_helper(len_to_idxs, False, True)

  def batches_helper(self, len_to_idxs, shuffle=True, test=False):
    # `len_to_idxs` summarizes sentence length to idx in `self.sents`.
    # This may be a subset of sentences, or full sentences.
    batches = []
    for length, idxs in len_to_idxs.items():
      if shuffle:
        idxs = np.random.permutation(idxs)

      def add_batch(begin, end):
        assert begin < end
        batches.append(idxs[begin:end])

      longest_sent_len = 0
      longest_action_len = 0
      batch_i = 0   # for i-th batch
      b = 0
      batch_token_size = self.batch_token_size
      batch_action_size = self.batch_action_size
      # Create each batch to guarantee that (batch_size*max_sent_len) does not exceed
      # batch_token_size.
      for i in range(len(idxs)):
        cur_sent_len = len(self.sents[idxs[i]].token_ids)
        cur_action_len = len(self.sents[idxs[i]].action_ids)
        longest_sent_len = max(longest_sent_len, cur_sent_len)
        longest_action_len = max(longest_action_len, cur_action_len)
        if len(self.sents[idxs[i]].token_ids) > 100:
          # Long sequence often requires larger memory and tend to cause memory error.
          # Here we try to reduce the elements in a batch for such sequences, considering
          # that they are rare and will not affect the total speed much.
          batch_token_size = int(self.batch_token_size * 0.7)
          batch_action_size = int(self.batch_action_size * 0.7)
        if (i > b and (  # for ensuring batch size 1
            (longest_sent_len * (batch_i+1) >= batch_token_size) or
            (longest_action_len * (batch_i+1) >= batch_action_size) or
            (batch_i > 0 and batch_i % self.batch_size == 0))):
          add_batch(b, i)
          batch_i = 0  # i is not included in prev batch
          longest_sent_len = cur_sent_len
          longest_action_len = cur_action_len
          b = i
          batch_token_size = self.batch_token_size
          batch_action_size = self.batch_action_size
        batch_i += 1
      add_batch(b, i+1)
    self.num_batches = len(batches)

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
      tokens = torch.tensor(self._pad_token_ids(token_ids), dtype=torch.long)
      ret = (tokens,)
      if not test:
        action_ids = [self.sents[i].action_ids for i in batch_idx]
        max_stack_size = max([self.sents[i].max_stack_size for i in batch_idx])
        ret += (torch.tensor(self._pad_action_ids(action_ids), dtype=torch.long),
                max_stack_size)
      if self.use_subwords:
        subword_end_mask = self.get_subword_end_mask(batch_idx)
      else:
        subword_end_mask = torch.full(tokens.size(), 1, dtype=torch.bool)
      ret += (subword_end_mask, batch_idx,)
      yield ret

  def get_subword_end_mask(self, sent_idxs):
    is_subword_ends = [self.sents[i].is_subword_end for i in sent_idxs]
    return torch.tensor(pad_items(is_subword_ends, 0)[0], dtype=torch.bool)

  def _get_len_to_idxs(self, sent_idxs = []):
    def to_len(token_ids):
      return len(token_ids)
    return self._get_len_to_idxs_helper(to_len, sent_idxs)

  def _get_grouped_len_to_idxs(self, sent_idxs = [], use_action_len = False, max_length_diff = 20):
    if use_action_len:
      def get_length(sent):
        return len(sent.action_ids)
    else:
      def get_length(sent):
        return len(sent.token_ids)

    if len(sent_idxs) == 0:
      sent_idxs = range(len(self.sents))
    len_to_idxs = defaultdict(list)
    group_size = self.group_sentence_size
    sent_idxs_with_len = sorted([(i, get_length(self.sents[i])) for i in sent_idxs], key=lambda x:x[1])
    b = 0
    last_idx = 0
    while b < len(sent_idxs_with_len):
      min_len = sent_idxs_with_len[b][1]
      max_len = sent_idxs_with_len[min(b+group_size, len(sent_idxs_with_len)-1)][1]
      if max_len - min_len < max_length_diff: # small difference in a group -> regist as a group
        group = [i for i, l in sent_idxs_with_len[b:b+group_size]]
        b += group_size
      else:
        e = b + 1
        while (e < len(sent_idxs_with_len) and
               sent_idxs_with_len[e][1] - min_len < max_length_diff):
          e += 1
        group = [i for i, l in sent_idxs_with_len[b:e]]
        b = e
      len_to_idxs[get_length(self.sents[group[-1]])] += group
    return len_to_idxs

  def _get_random_len_to_idxs(self, sent_idxs = []):
    def to_len(token_ids):
      return 1  # all sentences belong to the same group
    return self._get_len_to_idxs_helper(to_len, sent_idxs)

  def _get_len_to_idxs_helper(self, calc_len, sent_idxs = []):
    if len(sent_idxs) == 0:
      sent_idxs = range(len(self.sents))
    len_to_idxs = defaultdict(list)
    for i in sent_idxs:
      sent = self.sents[i]
      len_to_idxs[calc_len(sent.token_ids)].append(i)
    return len_to_idxs

  def _pad_action_ids(self, action_ids):
    action_ids, _ = pad_items(action_ids, self.action_dict.padding_idx)
    return action_ids

  def _pad_token_ids(self, token_ids):
    token_ids, _ = pad_items(token_ids, self.vocab.padding_idx)
    return token_ids

  def __len__(self):
    return self.num_batches
