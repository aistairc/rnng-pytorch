#!/usr/bin/env python3
import numpy as np
import itertools
import random
import torch
import torch.nn.functional as F
from nltk import Tree

def get_in_order_actions(line):
  def get_actions_recur(tree, actions=[]):
    if len(tree) == 1:
      if isinstance(tree[0], str):  # preterminal
        actions.append('SHIFT')
      else:  # unary
        actions = get_actions_recur(tree[0], actions)
        actions.append('NT({})'.format(tree.label()))
        actions.append('REDUCE')
    else:  # multiple children
      left_corner, others = tree[0], tree[1:]
      actions = get_actions_recur(left_corner, actions)
      actions.append('NT({})'.format(tree.label()))
      for c in others:
        actions = get_actions_recur(c, actions)
      actions.append('REDUCE')
    return actions

  tree = Tree.fromstring(line.strip())
  return get_actions_recur(tree) + ['FINISH']


def get_top_down_max_stack_size(actions):
  stack = []
  max_size = 0
  for a in actions:
    if a == 'SHIFT':
      stack.append('w')
    elif a[:2] == 'NT':
      stack.append('(')
    elif a == 'REDUCE':
      while stack[-1] != '(':
        stack.pop()
      stack[-1] = 'w'
    max_size = max(max_size, len(stack))
  if len(stack) != 1:
    print(stack)
  assert len(stack) == 1
  return max_size

def get_in_order_max_stack_size(actions):
  stack = []
  max_size = 0
  for a in actions:
    if a == 'SHIFT':
      stack.append('w')
    elif a[:2] == 'NT':
      lc = stack.pop()
      stack.append('(')
      stack.append(lc)
    elif a == 'REDUCE':
      while stack[-1] != '(':
        stack.pop()
      stack[-1] = 'w'
    max_size = max(max_size, len(stack))
  assert len(stack) == 1
  return max_size

def get_tree(actions, sent = None, SHIFT = 0, REDUCE = 1):
  #input action and sent (lists), e.g. S S R S S R R, A B C D
  #output tree ((A B) (C D))
  stack = []
  pointer = 0
  if sent is None:
    sent = list(map(str, range((len(actions)+1) // 2)))
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      stack.append('(' + left + ' ' + right + ')')
  assert(len(stack) == 1)
  return stack[-1]
      
def get_spans(actions, SHIFT = 0, REDUCE = 1):
  sent = list(range((len(actions)+1) // 2))
  spans = []
  pointer = 0
  stack = []
  for action in actions:
    if action == SHIFT:
      word = sent[pointer]
      stack.append(word)
      pointer += 1
    elif action == REDUCE:
      right = stack.pop()
      left = stack.pop()
      if isinstance(left, int):
        left = (left, None)
      if isinstance(right, int):
        right = (None, right)
      new_span = (left[0], right[1])
      spans.append(new_span)
      stack.append(new_span)
  return spans

def get_stats(span1, span2):
  tp = 0
  fp = 0
  fn = 0
  for span in span1:
    if span in span2:
      tp += 1
    else:
      fp += 1
  for span in span2:
    if span not in span1:
      fn += 1
  return tp, fp, fn

def update_stats(pred_span, gold_spans, stats):
  for gold_span, stat in zip(gold_spans, stats):
    tp, fp, fn = get_stats(pred_span, gold_span)
    stat[0] += tp
    stat[1] += fp
    stat[2] += fn

def get_f1(stats):
  f1s = []
  for stat in stats:
    prec = stat[0] / (stat[0] + stat[1])
    recall = stat[0] / (stat[0] + stat[2])
    f1 = 2*prec*recall / (prec + recall)*100 if prec+recall > 0 else 0.
    f1s.append(f1)
  return f1s


def span_str(start = None, end = None):
  assert(start is not None or end is not None)
  if start is None:
    return ' '  + str(end) + ')'
  elif end is None:
    return '(' + str(start) + ' '
  else:
    return ' (' + str(start) + ' ' + str(end) + ') '    


def get_tree_from_binary_matrix(matrix, length):    
  sent = list(map(str, range(length)))
  n = len(sent)
  tree = {}
  for i in range(n):
    tree[i] = sent[i]
  for k in np.arange(1, n):
    for s in np.arange(n):
      t = s + k
      if t > n-1:
        break
      if matrix[s][t].item() == 1:
        span = '(' + tree[s] + ' ' + tree[t] + ')'
        tree[s] = span
        tree[t] = span
  return tree[0]

def get_nonbinary_spans(actions, SHIFT = 0, REDUCE = 1):
  spans = []
  stack = []
  pointer = 0
  binary_actions = []
  nonbinary_actions = []
  num_shift = 0
  num_reduce = 0
  for action in actions:
    # print(action, stack)
    if action == "SHIFT":
      nonbinary_actions.append(SHIFT)
      stack.append((pointer, pointer))
      pointer += 1
      binary_actions.append(SHIFT)
      num_shift += 1
    elif action[:3] == 'NT(':
      stack.append('(')            
    elif action == "REDUCE":
      nonbinary_actions.append(REDUCE)
      right = stack.pop()
      left = right
      n = 1
      while stack[-1] is not '(':
        left = stack.pop()
        n += 1
      span = (left[0], right[1])
      if left[0] != right[1]:
        spans.append(span)
      stack.pop()
      stack.append(span)
      while n > 1:
        n -= 1
        binary_actions.append(REDUCE)        
        num_reduce += 1
    else:
      assert False  
  assert(len(stack) == 1)
  assert(num_shift == num_reduce + 1)
  return spans, binary_actions, nonbinary_actions

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

def length_to_mask(length):
  max_len = length.max()
  r = length.new_ones(length.size(0), max_len).cumsum(dim=1)
  return length.unsqueeze(1) >= r

def bincount_and_supply(x, max_size):
  counts = x.bincount()
  assert counts.size(0) <= max_size
  if counts.size(0) < max_size:
    counts = torch.cat([counts, counts.new_zeros(max_size - counts.size(0))])
  return counts

def masked_softmax(xs, mask, dim=-1):
  while mask.dim() < xs.dim():
    mask = mask.unsqueeze(1)
  e = xs.exp()
  e = e * mask
  return e / e.sum(dim=dim, keepdim=True)

def pad_items(items, pad_id):
  """
  `items`: a list of lists (each row has different number of elements).

  Return:
    padded_items: a converted items where shorter rows are padded by pad_id.
    lengths: lengths of rows in original items.
  """
  lengths = [len(row) for row in items]
  max_l = max(lengths)
  for i in range(len(items)):
    items[i] = items[i] + ([pad_id] * (max_l - len(items[i])))
  return items, lengths

def berkeley_unk_conv(ws):
  uk = "unk"
  sz = len(ws) - 1
  if ws[0].isupper():
      uk = "c" + uk
  if ws[0].isdigit() and ws[sz].isdigit():
      uk = uk + "n"
  elif sz <= 2:
      pass
  elif ws[sz-2:sz+1] == "ing":
      uk = uk + "ing"
  elif ws[sz-1:sz+1] == "ed":
      uk = uk + "ed"
  elif ws[sz-1:sz+1] == "ly":
      uk = uk + "ly"
  elif ws[sz] == "s":
      uk = uk + "s"
  elif ws[sz-2:sz+1] == "est":
      uk = uk + "est"
  elif ws[sz-1:sz+1] == "er":
      uk = uk + 'ER'
  elif ws[sz-2:sz+1] == "ion":
      uk = uk + "ion"
  elif ws[sz-2:sz+1] == "ory":
      uk = uk + "ory"
  elif ws[0:2] == "un":
      uk = "un" + uk
  elif ws[sz-1:sz+1] == "al":
      uk = uk + "al"
  else:
      for i in range(sz):
          if ws[i] == '-':
              uk = uk + "-"
              break
          elif ws[i] == '.':
              uk = uk + "."
              break
  return "<" + uk + ">"

