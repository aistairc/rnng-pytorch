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
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import numpy as np
import time
import logging
from data import Dataset
from models import TopDownRNNG
from in_order_models import InOrderRNNG
from fixed_stack_models import FixedStackRNNG
from fixed_stack_in_order_models import FixedStackInOrderRNNG
from utils import *

logger = logging.getLogger('train')

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-no-unk-train.json')
parser.add_argument('--val_file', default='data/ptb-no-unk-val.json')
parser.add_argument('--train_from', default='')
# Model options
parser.add_argument('--fixed_stack', action='store_true')
# parser.add_argument('--stack_size_bound', type=int, default=-1, help='If >0, stack size is bounded by this size. Training fails if some sentence requires larger stack sizes.')
parser.add_argument('--strategy', default='top_down', choices=['top_down', 'in_order'])
parser.add_argument('--w_dim', default=256, type=int, help='input/output word dimension')
parser.add_argument('--h_dim', default=256, type=int, help='LSTM hidden dimension')
parser.add_argument('--num_layers', default=2, type=int, help='number of layers in LM and the stack LSTM (for RNNG)')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--composition', default='lstm', choices=['lstm', 'attention'],
                    help='lstm: original lstm composition; attention: gated attention introduced in Kuncoro et al. (2017).')
parser.add_argument('--not_swap_in_order_stack', action='store_true',
                    help=('If True, prevent swapping elements by an open action for the in-order system.'
                          'WARNING: when --fixed_stack is True, this option is automatically and always set to True (obsolete option and no need to care)'))
# Optimization options
parser.add_argument('--batch_group', choices=['same_length', 'random', 'similar_length'],
                    default='similar_length', help='Sentences are grouped by this criterion to make each batch.')
parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam'], help='Which optimizer to use.')
parser.add_argument('--random_unk', action='store_true', help='Randomly replace a token to <unk> on training sentences by a probability inversely proportional to word frequency.')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--batch_token_size', type=int, default=15000, help='Number of tokens in a batch (batch_size*sentence_length) does not exceed this.')
parser.add_argument('--save_path', default='rnng.pt', help='where to save the best model')
parser.add_argument('--num_epochs', default=18, type=int, help='number of training epochs')
parser.add_argument('--min_epochs', default=8, type=int, help='do not decay learning rate for at least this many epochs')
parser.add_argument('--decay_cond_epochs', default=1, type=int, help='decay learning rate if loss does not improve conscutively this many steps')
#parser.add_argument('--mode', default='unsupervised', type=str, choices=['unsupervised', 'supervised'])
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--loss_normalize', default='batch', choices=['sum', 'batch', 'action'])
parser.add_argument('--decay', default=0.5, type=float, help='')
parser.add_argument('--param_init', default=0, type=float, help='parameter initialization (over uniform)')
parser.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=500, help='print stats after this many batches')
parser.add_argument('--valid_every', type=int, default=-1, help='If > 0, validate and save model every this many batches')
parser.add_argument('--tensorboard_log_dir', default='',
                    help='If not empty, tensor board summaries are recorded on the directory `tensor_board_log_dir/save_path`')
parser.add_argument('--amp', action='store_true')

class TensorBoardLogger(object):
  def __init__(self, args):
    if len(args.tensorboard_log_dir) > 0:
      log_dir = os.path.join(args.tensorboard_log_dir, args.save_path)
      self.writer = SummaryWriter(log_dir = log_dir)
      self.global_step = 0
      self.start_time = time.time()
    else:
      self.writer = None

  def write(self, kvs = {}, step = None, use_time=False):
    if self.writer is not None:
      if use_time:
        step = (time.time() - self.start_time)
      else:
        if step is None:
          step = self.global_step
          self.global_step += 1
      for k, v in kvs.items():
        self.writer.add_scalar(k, v, global_step=step)

def create_model(args, action_dict, vocab):
  model_args = {'action_dict': action_dict,
                'vocab': vocab.size(),
                'padding_idx': vocab.padding_idx,
                'w_dim': args.w_dim,
                'h_dim': args.h_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'attention_composition': args.composition == 'attention'}

  if args.strategy == 'top_down':
    if args.fixed_stack:
      model = FixedStackRNNG(**model_args)
    else:
      model = TopDownRNNG(**model_args)
  elif args.strategy == 'in_order':
    if args.fixed_stack:
      model = FixedStackInOrderRNNG(**model_args)
    else:
      model_args['do_swap_in_rnn'] = not args.not_swap_in_order_stack
      model = InOrderRNNG(**model_args)
  if args.param_init > 0:
    for param in model.parameters():
      param.data.uniform_(-args.param_init, args.param_init)
  return model

def main(args):
  logger.info('Args: {}'.format(args))
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  train_data = Dataset.from_json(args.train_file, args.batch_size, random_unk=args.random_unk,
                                 oracle=args.strategy, batch_group=args.batch_group,
                                 batch_token_size=args.batch_token_size)
  vocab = train_data.vocab
  action_dict = train_data.action_dict
  val_data = Dataset.from_json(args.val_file, args.batch_size, vocab, action_dict,
                               oracle=args.strategy)
  vocab_size = int(train_data.vocab_size)
  logger.info('Train: %d sents / %d batches, Val: %d sents / %d batches' %
              (len(train_data.sents), len(train_data), len(val_data.sents),
               len(val_data)))
  logger.info('Vocab size: %d' % vocab_size)
  cuda.set_device(args.gpu)
  if args.train_from == '':
    model = create_model(args, action_dict, vocab)
  else:
    logger.info('loading model from ' + args.train_from)
    checkpoint = torch.load(args.train_from)
    model = checkpoint['model']
  logger.info("model architecture")
  logger.info(model)
  if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.min_epochs = args.num_epochs
  model.train()
  model.cuda()

  epoch = 0
  decay= 0
  best_val_loss = 5e10
  val_losses = []

  if args.amp:
    scaler = torch.cuda.amp.GradScaler()

  global_batch_i = 0
  tb = TensorBoardLogger(args)
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

    def output_learn_log():
      param_norm = sum([p.norm()**2 for p in model.parameters()]).item()**0.5
      total_ll = total_a_ll + total_w_ll
      ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
      ll_diff =  total_ll - prev_ll
      action_ppl = np.exp(-total_a_ll / num_actions)
      word_ppl = np.exp(-total_w_ll / num_words)
      logger.info('Epoch: {}, Batch: {}/{}, LR: {:.4f}, '
                  'ActionPPL: {:.2f}, WordPPL: {:.2f}, '
                  'PPL: {:2f}, LL: {}, '
                  '|Param|: {:.2f}, Throughput: {:.2f} examples/sec'.format(
                    epoch, b, len(train_data), args.lr,
                    action_ppl, word_ppl,
                    ppl, -ll_diff,
                    param_norm, num_sents / (time.time() - start_time)
                  ))
      return ppl, word_ppl, action_ppl

    for batch in train_data.batches():
      token_ids, action_ids, max_stack_size, batch_idx = batch
      # print(max_stack_size, token_ids.size())
      token_ids = token_ids.cuda()
      action_ids = action_ids.cuda()
      b += 1
      global_batch_i += 1
      optimizer.zero_grad()

      def calc_loss():
        loss, a_loss, w_loss, _ = model(token_ids, action_ids, stack_size_bound=max_stack_size)
        if args.loss_normalize == 'sum':
          loss = loss
        elif args.loss_normalize == 'batch':
          loss = loss / token_ids.size(0)
        elif args.loss_normalize == 'action':
          loss = loss / a_loss.size(0)
        return loss, a_loss, w_loss

      if args.amp:
        with torch.cuda.amp.autocast():
          loss, a_loss, w_loss = calc_loss()
          scaler.scale(loss).backward()
        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
      else:
        loss, a_loss, w_loss = calc_loss()
        loss.backward()
        if args.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
      total_a_ll += -a_loss.sum().detach().item()
      total_w_ll += -w_loss.sum().detach().item()

      num_sents += token_ids.size(0)
      # assert token_ids.size(0) * token_ids.size(1) == w_loss.size(0)
      num_words += w_loss.size(0)
      num_actions += a_loss.size(0)

      # trying to obtain a discrete value that would have meaning at each epoch boundary.
      continuous_epoch = int(((epoch-1) + (b / len(train_data))) * 10000)

      if b % args.print_every == 0:
        ppl, word_ppl, _ = output_learn_log()
        prev_ll = total_a_ll + total_w_ll
        tb.write({'Train ppl': ppl, 'Train word ppl': word_ppl, 'lr': args.lr}, continuous_epoch)

      if args.valid_every > 0 and global_batch_i % args.valid_every == 0:
        do_valid(model, optimizer, train_data, val_data, tb, epoch, continuous_epoch, val_losses, args)

    output_learn_log()
    if args.valid_every <= 0:
      do_valid(model, optimizer, train_data, val_data, tb, epoch, epoch, val_losses, args)

  # Last validation is necessary when validations were performed intermediately.
  if args.valid_every > 0:
    do_valid(model, optimizer, train_data, val_data, tb, epoch, continuous_epoch, val_losses, args)
  logger.info("Finished training!")

def do_valid(model, optimizer, train_data, val_data, tb, epoch, step, val_losses, args):
  best_val_loss = float('inf') if len(val_losses) == 0 else min(val_losses)
  logger.info('--------------------------------')
  logger.info('Checking validation perplexity...')
  val_loss, val_ppl, val_action_ppl, val_word_ppl = eval_action_ppl(val_data, model)
  tb.write({'Valid ppl': val_ppl, 'Valid action ppl': val_action_ppl, 'Valid word ppl': val_word_ppl}, step)
  tb.write({'Valid loss': val_loss}, use_time=True)
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

  def consecutive_increase(target_losses):
    return (len(target_losses) > 0 and
            all(target_losses[i] < target_losses[i+1] for i in range(0, len(target_losses)-1)))

  val_losses.append(val_loss)
  if epoch > args.min_epochs and consecutive_increase(val_losses[-args.decay_cond_epochs-1:]):
    args.lr = args.decay*args.lr
    for param_group in optimizer.param_groups:
      param_group['lr'] = args.lr

def eval_action_ppl(data, model):
  model.eval()
  num_sents = 0
  num_words = 0
  num_actions = 0
  total_a_ll = 0
  total_w_ll = 0
  with torch.no_grad():
    for batch in data.batches():
      token_ids, action_ids, max_stack_size, batch_idx = batch
      token_ids = token_ids.cuda()
      action_ids = action_ids.cuda()
      loss, a_loss, w_loss, _ = model(token_ids, action_ids, stack_size_bound=max_stack_size)
      total_a_ll += -a_loss.sum().detach().item()
      total_w_ll += -w_loss.sum().detach().item()

      num_sents += token_ids.size(0)
      num_words += w_loss.size(0)
      num_actions += a_loss.size(0)

  ppl = np.exp((-total_a_ll - total_w_ll) / (num_actions + num_words))
  loss = -(total_a_ll + total_w_ll)
  action_ppl = np.exp(-total_a_ll / num_actions)
  word_ppl = np.exp(-total_w_ll / num_words)
  logger.info('PPL: {:2f}, Loss: {:2f}, ActionPPL: {:2f}, WordPPL: {:2f}'.format(
    ppl, loss, action_ppl, word_ppl))

  model.train()
  return loss, ppl, action_ppl, word_ppl

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
