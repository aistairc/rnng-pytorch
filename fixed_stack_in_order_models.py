import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from fixed_stack_models import FixedStack, FixedStackRNNG, TopDownState

class FixedInOrderStack(FixedStack):
  def __init__(self, initial_hidden, stack_size, input_size):
    super(FixedInOrderStack, self).__init__(initial_hidden, stack_size, input_size)

  def do_nt(self, nt_batches, nt_embs, nt_ids):
    left_corner = self.trees[nt_batches + (self.top_position[nt_batches]-1,)]
    self.trees[nt_batches + (self.top_position[nt_batches]-1,)] = nt_embs
    self.trees[nt_batches + (self.top_position[nt_batches],)] = left_corner

    self.nt_index_pos[nt_batches] = self.nt_index_pos[nt_batches] + 1
    self.nt_ids[nt_batches + (self.nt_index_pos[nt_batches],)] = nt_ids
    self.top_position[nt_batches] = self.top_position[nt_batches] + 1
    self.nt_index[nt_batches + (self.nt_index_pos[nt_batches],)] = self.top_position[nt_batches] - 1

class FixedStackInOrderRNNG(FixedStackRNNG):
  def __init__(self, action_dict,
               vocab = 100,
               padding_idx = 0,
               w_dim = 20,
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               attention_composition = False,
               max_open_nts = 100,
               max_cons_nts = 3,
  ):
    super(FixedStackInOrderRNNG, self).__init__(action_dict, vocab, padding_idx,
                                      w_dim, h_dim, num_layers,
                                      dropout, attention_composition,
                                      max_open_nts, max_cons_nts)

  def build_stack(self, x, batch_size = None):
    stack_size = max(150, x.size(1) + 100)
    batch_size = batch_size or x.size(0)
    initial_hidden = self.rnng.get_initial_hidden(x)
    return FixedInOrderStack(initial_hidden, stack_size, self.input_size)

  def initial_states(self, x, initial_stack = None):
    initial_hidden = self.rnng.get_initial_hidden(x)  # [(batch_size, hidden_size), (batch_size, hidden_size)]
    return [InOrderState.from_initial_stack((initial_hidden[0][b], initial_hidden[1][b]))
            for b in range(x.size(0))]

  def valid_action_mask(self, items, sent_len):
    mask = torch.ones((len(items), self.num_actions), dtype=torch.uint8)
    mask[:, self.action_dict.padding_idx] = 0
    for b, item in enumerate(items):
      state = item.state
      prev_action = item.action
      if state.finished():
        mask[b, :] = 0
        continue
      can_finish = state.pointer == sent_len and state.nopen_parens == 0
      if not can_finish:
        self.action_dict.mask_finish(mask, b)
      if state.pointer == sent_len:
        self.action_dict.mask_shift(mask, b)
      if state.nopen_parens == 0:
        self.action_dict.mask_reduce(mask, b)
        if state.pointer > 0:
          # Only stack element is left-corner.
          # Its parent has to be predicted immediately (only nt is valid).
          self.action_dict.mask_shift(mask, b)

      if len(state.hiddens) == 1 or self.action_dict.is_nt(prev_action):
        # successive nts is prohibited; it may lead to a valid parse, but
        # the same structure can always be achieved by finishing a left subtree,
        # followed by that nt.
        self.action_dict.mask_nt(mask, b)

      if state.nopen_parens > self.max_open_nts-1 or state.ncons_nts > self.max_cons_nts-1:
        # For in-order, cons_nts is accumuated by the loop of nt->reduce.
        # Except sentence final, we prohibit reduce to break this loop. Otherwise,
        # we fall into the state of one element in the stack, which prohibits following
        # shift (-> no way to escape).
        #
        # We instead prohibit nt for sentence final, because we need to close all
        # incomplete brackets.
        if state.pointer < sent_len:
          self.action_dict.mask_reduce(mask, b)
        else:
          self.action_dict.mask_nt(mask, b)

    mask = mask.to(items[0].state.hiddens[0][0][0].device)

    return mask

  def _is_last_action(self, action, state, shifted_all):
    return self.action_dict.is_finish(action)


class InOrderState(TopDownState):
  def __init__(self,
               pointer = 0,
               hiddens = None,
               cells = None,
               stack_trees = None,
               nopen_parens = 0,
               ncons_nts = 0,
               nt_index = None,
               nt_ids = None,
               prev_a = 0,
               is_finished = False):
    super(InOrderState, self).__init__(
      pointer, hiddens, cells, stack_trees, nopen_parens, ncons_nts, nt_index, nt_ids)

    self.prev_a = prev_a
    self.is_finished = is_finished

  def can_finish_by_reduce(self):
    # Ending by reduce is only valid for top-down so this action is only valid
    # for top-down.
    return False

  def can_finish_by_finish(self):
    return self.nopen_parens == 0

  def finished(self):
    return self.is_finished

  def copy(self):
    return InOrderState(self.pointer, self.hiddens[:], self.cells[:], self.stack_trees[:],
                        self.nopen_parens, self.ncons_nts, self.nt_index[:],
                        self.nt_ids[:], self.prev_a, self.is_finished)

  def do_action(self, a, action_dict):
    if action_dict.is_shift(a):
      self.pointer += 1
      self.ncons_nts = 0
    elif action_dict.is_nt(a):
      nt_id = action_dict.nt_id(a)
      self.nopen_parens += 1
      self.ncons_nts += 1
      self.nt_index.append(len(self.hiddens) - 2)  # already swapped
      self.nt_ids.append(nt_id)
    elif action_dict.is_reduce(a):
      self.nopen_parens -= 1
      # To regard repetitive nt->reduce->nt->reduce ... as cons nts,
      # we don't reset ncons_nts if previous action is nt.
      self.ncons_nts = self.ncons_nts if action_dict.is_nt(self.prev_a) else 0
    elif action_dict.is_finish(a):
      self.is_finished = True

    self.prev_a = a

  def update_stack(self, new_stack_top, new_tree_elem, action, action_dict):
    self.hiddens.append(new_stack_top[0])
    self.cells.append(new_stack_top[1])
    if action_dict.is_nt(action):
      left_corner = self.stack_trees.pop()
      self.stack_trees.extend([new_tree_elem, left_corner])
    else:
      self.stack_trees.append(new_tree_elem)


