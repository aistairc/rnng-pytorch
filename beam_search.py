import sys
import os

import argparse
import json
import random
import shutil
import copy
from collections import OrderedDict

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from in_order_models import InOrderRNNG
from fixed_stack_in_order_models import FixedStackInOrderRNNG

logger = logging.getLogger('train')

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--test_file', default='data/ptb-test.raw.txt')
parser.add_argument('--lm_output_file', default='surprisals.txt')
parser.add_argument('--model_file', default='rnng.pt')
parser.add_argument('--beam_size', type=int, default=200)
parser.add_argument('--word_beam_size', type=int, default=20)
parser.add_argument('--shift_size', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10, help='Please decrease this if memory error occurs.')
parser.add_argument('--block_size', type=int, default=100)
parser.add_argument('--batch_token_size', type=int, default=300,
                    help='Number of tokens in a batch (batch_size*sentence_length) does not exceed this. This value could be large value (e.g., 10000) safely when --stack_size_bound is set to some value > 0. Otherwise, we need to control (decrease) the batch size for longer sentences using this option, because then stack_size_bound will grow by sentence length.')
parser.add_argument('--stack_size_bound', type=int, default=-1,
                    help='Stack size during search is bounded by this size. If <= 0, the maximum size grows by sentence length (set by `sentence_length+10`). 100 looks sufficient for most of the grammars. Bounding to some value (e.g., 100) is useful to reduce memory usage while increasing beam size.')
parser.add_argument('--delay_word_ll', action='store_true',
                    help='Adding shift word probability is delayed at word-synchronizing step')
parser.add_argument('--particle_filter', action='store_true', help='search with particle filter')
parser.add_argument('--particle_size', type=int, default=10000)
parser.add_argument('--original_reweight', action='store_true',
                    help='If True, use the original reweighting (Eq. 4 in Crabbé et al. 2019) for particle filtering.')
parser.add_argument('--dump_beam', action='store_true',
                    help='(For debug and model development) print out all states in the beam at each step')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
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

def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  checkpoint = torch.load(args.model_file)
  vocab = checkpoint['vocab']
  action_dict = checkpoint['action_dict']
  prepro_args = checkpoint['prepro_args']
  model = load_model(checkpoint, action_dict, vocab)

  if args.fp16:
    model.half()

  # todo: add tagger.
  dataset = Dataset.from_text_file(args.test_file, args.batch_size, vocab, action_dict,
                                   prepro_args = prepro_args, batch_token_size = args.batch_token_size)
  logger.info("model architecture")
  logger.info(model)
  cuda.set_device(args.gpu)
  model.cuda()
  model.eval()

  if isinstance(model, InOrderRNNG) or isinstance(model, FixedStackInOrderRNNG):
    # A crude way to modify this parameter, which is set at training, and was previosuly defaulted to 8.
    # But I noticed that for in-order models, 8 is too much, and it considerably slows down the search.
    # In practice, for in-order models, max_cons_nts equals max length of unary chains, which, in PTB,
    # does not exceed 3, though this may be corpus or language specific. Here, I reset it to 3 manually.
    # For future, if all models are trained on the modified default values (now, in 3), this line could
    # be deleted.
    model.max_cons_nts = 3

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

  def dump_histroy(beam_history, sents):
    for pointer, bucket_i, beam, word_completed in beam_history:
      print('pointer: {}, i: {}'.format(pointer, bucket_i))
      for batch_i in range(len(beam)):
        for b in beam[batch_i]:
          print('beam: b_i: {}, {}'.format(batch_i, b.dump(action_dict, sents[batch_i])))
        for b in word_completed[batch_i]:
          print('word_comp: b_i: {}, {}'.format(batch_i, b.dump(action_dict, sents[batch_i])))
        print()

  if args.particle_filter:
    def parse(model, tokens, return_beam_history=False):
      return model.variable_beam_search(tokens, args.particle_size, args.original_reweight,
                                        stack_size_bound=args.stack_size_bound)
  else:
    def parse(model, tokens, return_beam_history=False):
      return model.word_sync_beam_search(
        tokens, args.beam_size, args.word_beam_size, args.shift_size,
        return_beam_history=return_beam_history,
        stack_size_bound=args.stack_size_bound)

  start_time = time.time()
  with torch.no_grad():

    block_idxs = []
    block_parses = []
    block_surprisals = []
    batches = [batch for batch in dataset.test_batches(args.block_size)]

    for batch in tqdm(batches):
      tokens, batch_idx = batch
      tokens = tokens.cuda()
      if args.dump_beam:
        parses, surprisals, beam_history = parse(model, tokens, True)
        dump_histroy(beam_history, [dataset.sents[idx] for idx in batch_idx])
      else:
        parses, surprisals = parse(model, tokens, False)

      best_actions = [p[0][0] for p in parses]  # p[0][1] is likelihood
      trees = [action_dict.build_tree_str(best_actions[i],
                                          dataset.sents[batch_idx[i]].orig_tokens,
                                          dataset.sents[batch_idx[i]].tags)
               for i in range(len(batch_idx))]
      block_idxs.extend(batch_idx)
      block_parses.extend(trees)
      block_surprisals.extend(surprisals)
      cur_block_size += tokens.size(0)

      if cur_block_size >= args.block_size:
        assert cur_block_size == args.block_size
        sort_and_print_trees(block_idxs, block_parses, block_surprisals)
        block_idxs = []
        block_parses = []
        block_surprisals = []
        cur_block_size = 0

  sort_and_print_trees(block_idxs, block_parses, block_surprisals)
  end_time = time.time()

  with open(args.lm_output_file, 'wt') as o:
    for sent_i, (sent, surp) in enumerate(zip(dataset.sents, all_surprisals)):
      orig_tokens = sent.orig_tokens
      input_tokens = [vocab.i2w[t_id] for t_id in sent.token_ids]
      assert len(orig_tokens) == len(surp)
      for t_i, (orig_t, mod_t, s) in enumerate(zip(orig_tokens, input_tokens, surp)):
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(sent_i, t_i, orig_t, mod_t, s))
    o.write('-----------------------------------\n')

    ll = -sum([sum(surp) for surp in all_surprisals])
    num_words = sum([len(surp) for surp in all_surprisals])
    ppl = np.exp(-ll / num_words)
    o.write('perplexity: {} Time: {} Throughput: {}'.format(
      ppl, end_time - start_time, (len(dataset.sents)) / (end_time-start_time)))

if __name__ == '__main__':
  args = parser.parse_args()
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
  )

  main(args)
