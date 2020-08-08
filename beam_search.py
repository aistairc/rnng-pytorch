import sys
import os

import argparse
import json
import random
import shutil
import copy

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from models import RNNG
from utils import *

logger = logging.getLogger('train')

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.raw.txt')
parser.add_argument('--lm_output_file', default='surprisals.txt')
parser.add_argument('--model_file', default='rnng.pt')
parser.add_argument('--beam_size', type=int, default=200)
parser.add_argument('--word_beam_size', type=int, default=20)
parser.add_argument('--shift_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--block_size', type=int, default=100)
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int)


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  checkpoint = torch.load(args.model_file)
  model = checkpoint['model']
  vocab = checkpoint['vocab']
  action_dict = checkpoint['action_dict']
  prepro_args = checkpoint['prepro_args']

  # todo: add tagger.
  dataset = Dataset.from_text_file(args.test_file, args.batch_size, vocab, action_dict,
                                   prepro_args = prepro_args)
  logger.info("model architecture")
  logger.info(model)
  cuda.set_device(args.gpu)
  model.cuda()
  model.eval()

  cur_block_size = 0

  all_parses = []
  all_surprisals = []

  def sort_and_print_trees(block_idxs, block_parses, block_surprisals):
    parse_idx_to_sent_idx = sorted(list(enumerate(block_idxs)), key=lambda x:x[1])
    orig_order_parses = [block_parses[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx]
    orig_order_surps = [block_surprisals[parse_idx] for (parse_idx, _) in parse_idx_to_sent_idx]

    all_parses.extend(orig_order_parses)
    all_surprisals.extend(orig_order_surps)

    for parse in orig_order_parses:
      print(parse)

  with torch.no_grad():

    block_idxs = []
    block_parses = []
    block_surprisals = []
    for batch in dataset.test_batches(args.block_size):
      tokens, batch_idx = batch
      tokens = tokens.cuda()
      parses, surprisals = model.word_sync_beam_search(
        tokens, args.beam_size, args.word_beam_size, args.shift_size)

      best_actions = [p[0][0] for p in parses]  # p[0][1] is likelihood
      trees = [action_dict.build_tree_str(best_actions[i],
                                          dataset.sents[batch_idx[i]].orig_tokens,
                                          dataset.sents[batch_idx[i]].tags)
               for i in range(len(batch_idx))]
      block_idxs.extend(batch_idx)
      block_parses.extend(trees)
      block_surprisals.extend(surprisals)
      cur_block_size += tokens.size(0)

      assert all(len(s) == tokens.size(1) for s in surprisals)

      if cur_block_size >= args.block_size:
        assert cur_block_size == args.block_size
        sort_and_print_trees(block_idxs, block_parses, block_surprisals)
        block_idxs = []
        block_parses = []
        block_surprisals = []
        cur_block_size = 0

  sort_and_print_trees(block_idxs, block_parses, block_surprisals)

  with open(args.lm_output_file, 'wt') as o:
    for sent_i, (sent, surp) in enumerate(zip(dataset.sents, all_surprisals)):
      orig_tokens = sent.orig_tokens
      input_tokens = [vocab.i2w[t_id] for t_id in sent.token_ids]
      assert len(orig_tokens) == len(surp)
      for t_i, (orig_t, mod_t, s) in enumerate(zip(orig_tokens, input_tokens, surp)):
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(sent_i, t_i, orig_t, mod_t, s))
    o.write('-----------------------------------\n')

    ll = -sum([sum(surp) for surp in surprisals])
    num_words = sum([len(surp) for surp in surprisals])
    ppl = np.exp(-ll / num_words)
    o.write('perplexity: {}'.format(ppl))

if __name__ == '__main__':
  args = parser.parse_args()
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
  )

  main(args)
