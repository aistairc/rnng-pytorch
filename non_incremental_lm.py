import sys
import os

import argparse
import copy
from collections import OrderedDict
from multiprocessing import Pool

import torch
from tqdm import tqdm

import numpy as np
import time
import logging
from data import Dataset, SentencePieceVocabulary, Sentence
from utils import *
from in_order_models import InOrderRNNG
from fixed_stack_in_order_models import FixedStackInOrderRNNG
from action_dict import TopDownActionDict, InOrderActionDict
from preprocess import get_sent_info

"""Evaluate joint likelihood for sampled parse trees.

The purpose of this script is to perform non-incremental decoding for trained RNNGs
(with importance sampling) as in the original decoding method in Dyer et al. (2016).

This script is not self-contained. What this script is doing is just one step among
multiple steps for decoding with a general RNNG, described below:
https://github.com/clab/rnng#decoding-with-the-generative-model

Specifically, this script performs the likelihood-evaluation step:
https://github.com/clab/rnng#evaluate-joint-likelihood-under-generative-model

Due to some difference in preprocessing, some additional preprocessing step is
necessary for the input sampled tree file. Assuming you have a sampled tree
file outputted by a trained discriminative parser (`test-samples.props`).

First, apply `rnng/utils/add-fake-preterms-for-eval.pl` to this file:
> perl rnng/utils/add-fake-preterms-for-eval.pl test-samples.props > test-samples.props.tagged

Here, we assume DyNet `rnng` is downloaded in this directory.

Then, run the second step to extract the row for sampled trees:
> rnng/utils/cut-corpus.pl 3 test-samples.props.tagged > test-samples.trees

The last conversion step is to remove UNK-ed tokens from the sampled trees. Since
rnng-pytorch does perform preprocessing for unks (including subwords) internally,
all unknown words should be replaced with the original tokens. To do this,
run the command:
> scripts/remove_unk_from_samples.py [original input sentence file] test-samples.trees [samples] > test-samples.trees.no_unk

This output becomes the input (`--test_file`) for this script.
> python non_incremental_lm.py --test_file test-samples.trees.no_unk --model_file [rnng.pt] > test-samples.likelihoods

This output can be used as an input to the final step for LM evaluation:
https://github.com/clab/rnng#estimate-marginal-likelihood-final-step-to-get-language-modeling-ppl

Note that this decoding with importance sampling does not allow calculating
token-level surprisals (differently from beam search). We can obtain sentence-level
log-likelihood only.
"""

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='test-samples.trees')
parser.add_argument('--model_file', default='rnng.pt')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='If "cuda", GPU number --gpu is used.')
parser.add_argument('--seed', default=3435, type=int)
parser.add_argument('--fp16', action='store_true')

def load_model(checkpoint, action_dict, vocab):
  if 'model_state_dict' in checkpoint:
    from train import create_model
    model = create_model(checkpoint['args'], action_dict, vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
  else:
    return checkpoint['model']

def create_dataset(vocab, action_dict, args):
  if isinstance(vocab, SentencePieceVocabulary):
    is_subword = True
    sp = vocab.sp
  else:
    is_subword = False
    sp = None

  if isinstance(action_dict, TopDownActionDict):
    oracle = 'top_down'
  elif isinstance(action_dict, InOrderActionDict):
    oracle = 'in_order'
  nonterminals = [a[3:-1] for a in action_dict.i2a[action_dict.nt_begin_id():]]
  # These are dummy action dicts, which are necessary to properly run get_sent_info()
  # (in preprocess.py) below.
  top_down_action_dict = TopDownActionDict(nonterminals)
  in_order_action_dict = InOrderActionDict(nonterminals)

  sents = []
  with open(args.test_file, 'r') as in_f:
    tree_with_settings = []
    conv_setting = (False, False, vocab, sp, top_down_action_dict, in_order_action_dict)
    for tree in in_f:
      tree_with_settings.append((tree, conv_setting))
    with Pool() as pool:
      for sent_info in pool.map(get_sent_info, tree_with_settings):
        sents.append(Sentence.from_json(sent_info, oracle=oracle))
  _inf = float('inf')
  return Dataset(sents, args.batch_size, vocab, action_dict,
                 batch_token_size=_inf, batch_action_size=_inf, batch_group='random',
                 max_length_diff=_inf, group_sentence_size=_inf)

def get_len_from_padded(x, pad_id):
  return (x != pad_id).sum(1)

def accumulate_to_sent_loss(token_loss, sent_lens):
  offset = 0
  sent_loss = token_loss.new_zeros((sent_lens.size(0)))
  for i in range(sent_loss.size(0)):
    sent_loss[i] = token_loss[offset:offset+sent_lens[i]].sum()
    offset += sent_lens[i]
  return sent_loss

def main(args):
  if args.device == 'cuda':
    device = 'cuda:{}'.format(args.gpu)
  else:
    device = 'cpu'

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  checkpoint = torch.load(args.model_file)
  vocab = checkpoint['vocab']
  action_dict = checkpoint['action_dict']
  prepro_args = checkpoint['prepro_args']
  model = load_model(checkpoint, action_dict, vocab).to(device)

  if args.fp16:
    model.half()

  dataset = create_dataset(vocab, action_dict, args)
  logger.info("model architecture")
  logger.info(model)
  model.eval()

  all_parses = []
  all_surprisals = []

  start_time = time.time()
  with torch.no_grad():
    batches = [b for b in dataset.batches(shuffle=False)]
    for batch in tqdm(batches):
      token_ids, action_ids, max_stack_size, subword_end_mask, batch_idx = batch
      token_ids = token_ids.to(device)
      action_ids = action_ids.to(device)
      subword_end_mask = subword_end_mask.to(device)
      loss, a_loss, w_loss, _ = model(token_ids, action_ids,
                                      stack_size_bound=max_stack_size,
                                      subword_end_mask=subword_end_mask)

      token_ids = token_ids.cpu()
      action_ids = action_ids.cpu()

      token_len = get_len_from_padded(token_ids, vocab.padding_idx)
      action_len = get_len_from_padded(action_ids, action_dict.padding_idx)
      a_loss = a_loss.cpu()
      w_loss = w_loss.cpu()
      sent_loss = (accumulate_to_sent_loss(a_loss, action_len) +
                   accumulate_to_sent_loss(w_loss, token_len))

      for i in range(sent_loss.size(0)):
        orig_token_size = len([b for b in dataset.sents[batch_idx[i]].is_subword_end if b])
        print('{} {}'.format(orig_token_size, sent_loss[i]))

if __name__ == '__main__':
  args = parser.parse_args()
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
  )

  main(args)
