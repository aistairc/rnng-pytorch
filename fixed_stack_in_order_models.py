import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from fixed_stack_models import BeamItems, FixedStack, FixedStackRNNG, TopDownState, StackState

class FixedInOrderStack(FixedStack):
  def __init__(self, initial_hidden, stack_size, input_size, beam_size = 1):
    super(FixedInOrderStack, self).__init__(initial_hidden, stack_size, input_size, beam_size)

  def do_nt(self, nt_batches, nt_embs, nt_ids):
    left_corner = self.trees[nt_batches + (self.top_position[nt_batches]-1,)]
    self.trees[nt_batches + (self.top_position[nt_batches]-1,)] = nt_embs
    self.trees[nt_batches + (self.top_position[nt_batches],)] = left_corner

    self.nt_index_pos[nt_batches] = self.nt_index_pos[nt_batches] + 1
    self.nt_ids[nt_batches + (self.nt_index_pos[nt_batches],)] = nt_ids
    self.top_position[nt_batches] = self.top_position[nt_batches] + 1
    self.nt_index[nt_batches + (self.nt_index_pos[nt_batches],)] = self.top_position[nt_batches] - 1


class InOrderStackState(StackState):
  def __init__(self, batch_size, beam_size, device):
    super(InOrderStackState, self).__init__(batch_size, beam_size, device)

  def update_nt_counts(self, actions, action_dict, action_path):
    shift_idxs = (actions == action_dict.a2i['SHIFT']).nonzero(as_tuple=True)
    nt_idxs = (actions >= action_dict.nt_begin_id()).nonzero(as_tuple=True)
    reduce_idxs = (actions == action_dict.a2i['REDUCE']).nonzero(as_tuple=True)
    prev_is_not_nt = (action_path.prev_actions() < action_dict.nt_begin_id())

    self.ncons_nts[shift_idxs] = 0
    self.nopen_parens[nt_idxs] += 1
    self.ncons_nts[nt_idxs] += 1
    self.nopen_parens[reduce_idxs] -= 1

    # To regard repetitive nt->reduce->nt->reduce ... as cons nts,
    # we don't reset ncons_nts if previous action is nt.
    reset_ncons_reduce_mask = (actions == action_dict.a2i['REDUCE']) * (prev_is_not_nt)
    self.ncons_nts[reset_ncons_reduce_mask] = 0


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
    initial_hidden = self.rnng.get_initial_hidden(x)
    return FixedInOrderStack(initial_hidden, stack_size, self.input_size)

  def new_beam_stack_with_state(self, initial_hidden, stack_size, beam_size):
    stack = FixedInOrderStack(initial_hidden, stack_size, self.input_size, beam_size)
    stack_state = InOrderStackState(initial_hidden[0].size(0), beam_size, initial_hidden[0].device)
    return stack, stack_state

  def invalid_action_mask(self, beam, sent_len):
    action_order = torch.arange(self.num_actions, device=beam.nopen_parens.device)

    nopen_parens = beam.nopen_parens
    ncons_nts = beam.ncons_nts
    pointer = beam.stack.pointer
    top_position = beam.stack.top_position
    prev_actions = beam.prev_actions()

    # For in-order, cons_nts is accumuated by the loop of nt->reduce.
    # Except sentence final, we prohibit reduce to break this loop. Otherwise,
    # we fall into the state of one element in the stack, which prohibits following
    # shift (-> no way to escape).
    #
    # We instead prohibit nt for sentence final, because we need to close all
    # incomplete brackets.
    # This mask is a precondition used both for reduce_mask and reduce_nt
    # (depending on sentence final or not).
    pre_nt_reduce_mask = ((nopen_parens > self.max_open_nts-1) +
                          (ncons_nts > self.max_cons_nts-1))

    # reduce_mask[i,j,k]=True means k is a not allowed reduce action for (i,j).
    reduce_mask = (action_order == self.action_dict.a2i['REDUCE']).view(1, 1, -1)
    reduce_mask = (((nopen_parens == 0) +
                    (pre_nt_reduce_mask * (pointer < sent_len))).unsqueeze(-1) *
                   reduce_mask)

    finish_mask = (action_order == self.action_dict.finish_action()).view(1, 1, -1)
    finish_mask = (((pointer != sent_len) + (nopen_parens != 0)).unsqueeze(-1) *
                   finish_mask)

    shift_mask = (action_order == self.action_dict.a2i['SHIFT']).view(1, 1, -1)
    shift_mask = (((pointer == sent_len) +
                   ((pointer > 0) * (nopen_parens == 0)) +
                   # when nopen=0, shift accompanies nt, thus requires two.
                   ((nopen_parens == 0) * (top_position >= beam.stack.stack_size-2)) +
                   # otherwise, requires one room.
                   ((nopen_parens > 0) * (top_position >= beam.stack.stack_size-1))).unsqueeze(-1) *
                  shift_mask)

    nt_mask = (action_order >= self.action_dict.nt_begin_id()).view(1, 1, -1)
    prev_is_nt = prev_actions >= self.action_dict.nt_begin_id()
    nt_mask = ((prev_is_nt +
                (top_position == 0) +
                # prohibit nt version of pre_nt_reduce_mask (reduce is prohibited except empty buffer)
                (pre_nt_reduce_mask * (pointer == sent_len)) +
                # reduce is allowed after nt, so minimal room for stack is 1.
                (top_position >= beam.stack.stack_size-1) +
                # +1 for final finish.
                (beam.actions.size(2) - beam.actions_pos < (
                  sent_len - beam.stack.pointer + beam.nopen_parens + 1))).unsqueeze(-1) *
               nt_mask)

    pad_mask = (action_order == self.action_dict.padding_idx).view(1, 1, -1)
    finished_mask = ((prev_actions == self.action_dict.finish_action()) +
                     (prev_actions == self.action_dict.padding_idx)).unsqueeze(-1)
    beam_width_mask = (torch.arange(beam.beam_size, device=reduce_mask.device).unsqueeze(0) >=
                       beam.beam_widths.unsqueeze(1)).unsqueeze(-1)

    return (reduce_mask + finish_mask + shift_mask + nt_mask + pad_mask + finished_mask +
            beam_width_mask)

  def _parse_finish_mask(self, beam, action_id, beam_id):
    return action_id == self.action_dict.finish_action()
