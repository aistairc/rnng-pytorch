import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from models import TopDownRNNG, TopDownState

class InOrderRNNG(TopDownRNNG):
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
               do_swap_in_rnn = True,
  ):
    super(InOrderRNNG, self).__init__(action_dict, vocab, padding_idx,
                                      w_dim, h_dim, num_layers,
                                      dropout, attention_composition,
                                      max_open_nts, max_cons_nts)
    self.do_swap_in_rnn = do_swap_in_rnn

  def initial_states(self, x, initial_stack = None):
    initial_hs = self._initial_hs(x, initial_stack)
    return [InOrderState.from_initial_stack(h) for h in initial_hs]

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

      if len(state.stack) == 1 or self.action_dict.is_nt(prev_action):
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

    mask = mask.to(items[0].state.stack[0][0][0].device)

    return mask

  def update_stack_rnn(self, states, actions, shift_idx, shifted_embs):
    assert actions.size(0) == len(states)
    assert shift_idx.size(0) == shifted_embs.size(0)

    reduces = (actions == self.action_dict.a2i['REDUCE']).nonzero().squeeze(1)
    nts = (actions >= self.action_dict.nt_begin_id()).nonzero().squeeze(1)
    not_nts = (actions < self.action_dict.nt_begin_id()).nonzero().squeeze(1)

    new_stack_input = shifted_embs.new_zeros(actions.size(0), self.w_dim)

    if not_nts.size(0) > 0:
      if reduces.size(0) > 0:
        reduce_idx = reduces.cpu().numpy()
        reduce_states = [states[i] for i in reduce_idx]
        children, ch_lengths, nt, nt_id = self._collect_children_for_reduce(reduce_states)
        reduce_context = self.stack_top_h(reduce_states)
        reduce_context = self.stack_to_hidden(reduce_context)
        new_child, _, _ = self.composition(children, ch_lengths, nt, nt_id, reduce_context)
        new_stack_input[reduces] = new_child.float()

      new_stack_input[shift_idx] = shifted_embs
      not_nt_new_stack_input = new_stack_input[not_nts]
      not_nt_idx = not_nts.cpu().numpy()
      not_nt_stack_top_context = self._collect_stack_top_context(states, not_nt_idx)

      not_nt_new_stack_top = self.stack_rnn(not_nt_new_stack_input, not_nt_stack_top_context)
      for i, b in enumerate(not_nt_idx):
        new_stack_top_b = [[layer[0][i], layer[1][i]] for layer in not_nt_new_stack_top]
        states[b].update_stack(new_stack_top_b, not_nt_new_stack_input[i])

    if nts.size(0) > 0:
      nt_idx = nts.cpu().numpy()
      nt_ids = (actions[nts] - self.action_dict.nt_begin_id())
      nt_embs = self.nt_emb(nt_ids)  # new_stack_input for nt
      if self.do_swap_in_rnn:
        # For nt, current top stack is discarded; insert two elements top of the current
        # 2nd-top element.
        nt_stack_top2_context = self._collect_stack_top_context(states, nt_idx, 2)
        nt_stack_top_context = self.stack_rnn(nt_embs, nt_stack_top2_context)
        left_corners = torch.stack([states[i].stack_trees[-1] for i in nt_idx], 0)
        nt_new_stack_top = self.stack_rnn(left_corners, nt_stack_top_context)
      else:
        # state.stack is not swapped, only stack_trees is swapped.
        nt_stack_top_context = self._collect_stack_top_context(states, nt_idx, 1)
        nt_new_stack_top = self.stack_rnn(nt_embs, nt_stack_top_context)

      for i, b in enumerate(nt_idx):
        new_stack_tops_b = [[[layer[0][i], layer[1][i]] for layer in nt_stack_top_context],
                            [[layer[0][i], layer[1][i]] for layer in nt_new_stack_top]]
        states[b].update_stack_with_swap(new_stack_tops_b, nt_embs[i])
        assert len(states[b].stack) == len(states[b].stack_trees) + 1

    for b in range(len(states)):
      states[b].do_action(actions[b].item(), self.action_dict)

  def _is_last_action(self, action, state, shifted_all):
    return self.action_dict.is_finish(action)


class InOrderState(TopDownState):
  def __init__(self,
               pointer = 0,
               stack = None,
               stack_trees = None,
               nopen_parens = 0,
               ncons_nts = 0,
               nt_index = None,
               nt_ids = None,
               prev_a = 0,
               is_finished = False):
    super(InOrderState, self).__init__(
      pointer, stack, stack_trees, nopen_parens, ncons_nts, nt_index, nt_ids)

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
    return InOrderState(self.pointer, self.stack[:], self.stack_trees[:],
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
      self.nt_index.append(len(self.stack) - 2)  # already swapped
      self.nt_ids.append(nt_id)
    elif action_dict.is_reduce(a):
      self.nopen_parens -= 1
      # To regard repetitive nt->reduce->nt->reduce ... as cons nts,
      # we don't reset ncons_nts if previous action is nt.
      self.ncons_nts = self.ncons_nts if action_dict.is_nt(self.prev_a) else 0
    elif action_dict.is_finish(a):
      self.is_finished = True

    self.prev_a = a

  def update_stack_with_swap(self, new_stack_two_tops, new_tree_elem):
    self.stack.pop()
    self.stack.extend(new_stack_two_tops)
    left_corner = self.stack_trees.pop()
    self.stack_trees.extend([new_tree_elem, left_corner])

