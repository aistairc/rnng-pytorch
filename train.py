#!/usr/bin/env python3
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
from models import TopDownRNNG
from utils import *

logger = logging.getLogger('train')

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-no-unk-train.json')
parser.add_argument('--val_file', default='data/ptb-no-unk-val.json')
parser.add_argument('--train_from', default='')
# Model options
parser.add_argument('--w_dim', default=650, type=int, help='input/output word dimension')
parser.add_argument('--h_dim', default=650, type=int, help='LSTM hidden dimension')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in LM and the stack LSTM (for RNNG)')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--composition', default='lstm', choices=['lstm', 'attention'],
                    help='lstm: original lstm composition; attention: gated attention introduced in Kuncoro et al. (2017).')
# Optimization options
parser.add_argument('--no_random_unk', action='store_true', help='Prohibit to randomly replace a token to <unk> on training sentences (in default, randomly replace).')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--save_path', default='rnng.pt', help='where to save the data')
parser.add_argument('--num_epochs', default=18, type=int, help='number of training epochs')
parser.add_argument('--min_epochs', default=8, type=int, help='do not decay learning rate for at least this many epochs')
#parser.add_argument('--mode', default='unsupervised', type=str, choices=['unsupervised', 'supervised'])
parser.add_argument('--lr', default=1, type=float, help='starting learning rate')
parser.add_argument('--loss_normalize', default='batch', choices=['sum', 'batch', 'action'])
parser.add_argument('--decay', default=0.5, type=float, help='')
parser.add_argument('--param_init', default=0, type=float, help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=500, help='print stats after this many batches')

def main(args):
  logger.info('Args: {}'.format(args))
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  train_data = Dataset.from_json(args.train_file, args.batch_size, random_unk=not args.no_random_unk)
  vocab = train_data.vocab
  action_dict = train_data.action_dict
  val_data = Dataset.from_json(args.val_file, args.batch_size, vocab, action_dict)
  vocab_size = int(train_data.vocab_size)
  logger.info('Train: %d sents / %d batches, Val: %d sents / %d batches' %
              (len(train_data.sents), len(train_data), len(val_data.sents),
               len(val_data)))
  logger.info('Vocab size: %d' % vocab_size)
  cuda.set_device(args.gpu)
  if args.train_from == '':
    model = TopDownRNNG(train_data.action_dict,
                        vocab = vocab_size,
                        padding_idx = vocab.padding_idx,
                        w_dim = args.w_dim,
                        h_dim = args.h_dim,
                        num_layers = args.num_layers,
                        dropout = args.dropout,
                        attention_composition = args.composition == 'attention')
    if args.param_init > 0:
      for param in model.parameters():
        param.data.uniform_(-args.param_init, args.param_init)
  else:
    logger.info('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
  logger.info("model architecture")
  logger.info(model)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  model.train()
  model.cuda()

  epoch = 0
  decay= 0
  best_val_loss = 5e10
  # best_val_ppl, best_val_f1 = eval(val_data, model, samples = args.mc_samples,
  #                                  count_eos_ppl = args.count_eos_ppl)
  # all_stats = [[0., 0., 0.]] #true pos, false pos, false neg for f1 calc
  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1
    logger.info('Starting epoch {}'.format(epoch))
    num_sents = 0.
    num_words = 0.
    num_actions = 0
    b = 0
    total_a_ll = 0.
    total_w_ll = 0.
    prev_ll = 0.
    for batch in train_data.batches():
      token_ids, action_ids, batch_idx = batch
      token_ids = token_ids.cuda()
      action_ids = action_ids.cuda()
      b += 1
      optimizer.zero_grad()
      loss, a_loss, w_loss, _ = model(token_ids, action_ids)
      total_a_ll += -a_loss.sum().detach().item()
      total_w_ll += -w_loss.sum().detach().item()
      if args.loss_normalize == 'sum':
        loss = loss
      elif args.loss_normalize == 'batch':
        loss = loss / token_ids.size(0)
      elif args.loss_normalize == 'action':
        loss = loss / a_loss.size(0)
      loss.backward()
      if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      optimizer.step()
      num_sents += token_ids.size(0)
      assert token_ids.size(0) * token_ids.size(1) == w_loss.size(0)
      num_words += w_loss.size(0)
      num_actions += a_loss.size(0)

      if b % args.print_every == 0:
        param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
        total_ll = total_a_ll + total_w_ll
        ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
        ll_diff =  total_ll - prev_ll
        prev_ll = total_ll
        logger.info('Epoch: {}, Batch: {}/{}, LR: {:.4f}, '
                    'ActionPPL: {:.2f}, WordPPL: {:.2f}, '
                    'PPL: {:2f}, LL: {}, '
                    '|Param|: {:.2f}, Throughput: {:.2f} examples/sec'.format(
                      epoch, b, len(train_data), args.lr,
                      np.exp(-total_a_ll / num_actions),
                      np.exp(-total_w_ll / num_words),
                      ppl, -ll_diff,
                      param_norm, num_sents / (time.time() - start_time)
                    ))
    logger.info('--------------------------------')
    logger.info('Checking validation perplexity...')
    val_loss, val_ppl = eval_action_ppl(val_data, model)
    logger.info('--------------------------------')
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'vocab': train_data.vocab,
        'prepro_args': train_data.prepro_args,
        'action_dict': train_data.action_dict
      }
      logger.info('Saving checkpoint to {}'.format(args.save_path))
      torch.save(checkpoint, args.save_path)
      model.cuda()
    else:
      if epoch > args.min_epochs:
        decay = 1
      if decay == 1:
        args.lr = args.decay*args.lr
        for param_group in optimizer.param_groups:
          param_group['lr'] = args.lr
  logger.info("Finished training!")

def eval_action_ppl(data, model):
  model.eval()
  num_sents = 0
  num_words = 0
  num_actions = 0
  total_a_ll = 0
  total_w_ll = 0
  with torch.no_grad():
    for batch in data.batches():
      token_ids, action_ids, batch_idx = batch
      token_ids = token_ids.cuda()
      action_ids = action_ids.cuda()
      loss, a_loss, w_loss, _ = model(token_ids, action_ids)
      total_a_ll += -a_loss.sum().detach().item()
      total_w_ll += -w_loss.sum().detach().item()

      num_sents += token_ids.size(0)
      num_words += w_loss.size(0)
      num_actions += a_loss.size(0)

  ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
  loss = -(total_a_ll + total_w_ll)
  logger.info('PPL: {:2f}, Loss: {:2f}, ActionPPL: {:2f}, WordPPL: {:2f}'.format(
    ppl, loss, np.exp(-total_a_ll / num_actions), np.exp(-total_w_ll / num_words)))

  model.train()
  return loss, ppl

if __name__ == '__main__':
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
    handlers=[
      logging.FileHandler("{}.log".format(args.save_path)),
      logging.StreamHandler()
    ])

  main(args)
