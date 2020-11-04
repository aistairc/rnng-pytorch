import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import *
from torch.distributions import Bernoulli
import itertools


class MultiLayerLSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0):
    super(MultiLayerLSTMCell, self).__init__()
    self.lstm = nn.ModuleList()
    self.lstm.append(nn.LSTMCell(input_size, hidden_size))
    for i in range(num_layers-1):
      self.lstm.append(nn.LSTMCell(hidden_size, hidden_size))
    self.num_layers = num_layers
    self.dropout = dropout
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, input, prev):
    """

    :param input: (batch_size, input_size)
    :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
    """

    next_hidden = []
    next_cell = []

    if prev is None:
        prev = (
            input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0),
            input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0))

    for i in range(self.num_layers):
      prev_hidden_i = prev[0][:, :, i]
      prev_cell_i = prev[1][:, :, i]
      if i == 0:
        next_hidden_i, next_cell_i = self.lstm[i](input, (prev_hidden_i, prev_cell_i))
      else:
        input_im1 = self.dropout_layer(input_im1)
        next_hidden_i, next_cell_i = self.lstm[i](input_im1, (prev_hidden_i, prev_cell_i))
      next_hidden += [next_hidden_i]
      next_cell += [next_cell_i]
      input_im1 = next_hidden_i

    next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
    next_cell = torch.stack(next_cell).permute(1, 2, 0)
    return next_hidden, next_cell


class LSTMComposition(nn.Module):
  def __init__(self, dim, dropout):
    super(LSTMComposition, self).__init__()
    self.dim = dim
    self.rnn = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
    self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim*2, dim), nn.ReLU())

    self.batch_index = torch.arange(0, 100000, dtype=torch.long)  # cache with sufficient number.

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):
    """

    :param children: (batch_size, max_num_children, input_dim)
    :param ch_lengths: (batch_size)
    :param nt: (batch_size, input_dim)
    :param nt_id: (batch_size)
    """
    lengths = ch_lengths + 2
    nt = nt.unsqueeze(1)
    elems = torch.cat([nt, children, torch.zeros_like(nt)], dim=1)
    elems[self.batch_index[:elems.size(0)], lengths-1] = nt.squeeze(1)

    packed = pack_padded_sequence(elems, lengths.int().cpu(), batch_first=True, enforce_sorted=False)
    h, _ = self.rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True)

    gather_idx = (lengths - 2).unsqueeze(1).expand(-1, h.size(-1)).unsqueeze(1)
    fwd = h.gather(1, gather_idx).squeeze(1)[:, :self.dim]
    bwd = h[:, 1, self.dim:]
    c = torch.cat([fwd, bwd], dim=1)

    return self.output(c), None, None


class AttentionComposition(nn.Module):
  def __init__(self, w_dim, dropout, num_labels = 10):
    super(AttentionComposition, self).__init__()
    self.w_dim = w_dim
    self.num_labels = num_labels
    self.dropout = nn.Dropout(dropout)

    self.rnn = nn.LSTM(w_dim, w_dim, bidirectional=True, batch_first=True)

    self.V = nn.Linear(2*w_dim, 2*w_dim, bias=False)
    self.nt_emb = nn.Embedding(num_labels, w_dim)  # o_nt in the Kuncoro et al. (2017)
    self.nt_emb2 = nn.Sequential(nn.Embedding(num_labels, w_dim*2), self.dropout)  # t_nt in the Kuncoro et al. (2017)
    self.gate = nn.Sequential(nn.Linear(w_dim*4, w_dim*2), nn.Sigmoid())
    self.output = nn.Sequential(nn.Linear(w_dim*2, w_dim), nn.ReLU())

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):  # children: (batch_size, n_children, w_dim)

    packed = pack_padded_sequence(children, ch_lengths.int().cpu(), batch_first=True, enforce_sorted=False)
    h, _ = self.rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True)  # (batch, n_children, 2*w_dim)

    rhs = torch.cat([self.nt_emb(nt_id), stack_state], dim=1) # (batch_size, w_dim*2, 1)
    logit = (self.V(h)*rhs.unsqueeze(1)).sum(-1)  # equivalent to bmm(self.V(h), rhs.unsqueeze(-1)).squeeze(-1)
    len_mask = (ch_lengths.new_zeros(children.size(0), 1) +
                torch.arange(children.size(1), device=children.device)) >= ch_lengths.unsqueeze(1)
    logit[len_mask] = -float('inf')

    attn = F.softmax(logit, -1)
    weighted_child = (h*attn.unsqueeze(-1)).sum(1)
    weighted_child = self.dropout(weighted_child)

    nt2 = self.nt_emb2(nt_id)  # (batch_size, w_dim)
    gate_input = torch.cat([nt2, weighted_child], dim=-1)
    g = self.gate(gate_input)  # (batch_size, w_dim)
    c = g*nt2 + (1-g)*weighted_child  # (batch_size, w_dim)

    return self.output(c), attn, g


class FixedStack:
  def __init__(self, initial_hidden, stack_size, input_size, beam_size = 1):
    super(FixedStack, self).__init__()
    device = initial_hidden[1].device
    hidden_size = initial_hidden[0].size(-2)
    num_layers = initial_hidden[0].size(-1)

    if beam_size == 1:
      batch_size = (initial_hidden[0].size(0),)
      self.batch_index = (torch.arange(0, batch_size[0], dtype=torch.long, device=device),)
    else:
      batch_size = (initial_hidden[0].size(0), beam_size)
      self.batch_index = ((torch.arange(0, batch_size[0], dtype=torch.long, device=device)
                           .unsqueeze(1).expand(-1, beam_size).reshape(-1)),
                          torch.cat([torch.arange(0, beam_size, dtype=torch.long, device=device)
                                     for _ in range(batch_size[0])]))

    self.batch_size = initial_hidden[0].size(0)
    self.beam_size = beam_size
    self.stack_size = stack_size

    self.pointer = torch.zeros(batch_size, dtype=torch.long, device=device)  # word pointer
    self.top_position = torch.zeros(batch_size, dtype=torch.long, device=device)  # stack top position
    self.hiddens = torch.zeros(batch_size + (stack_size+1, hidden_size, num_layers), device=device)
    self.cells = torch.zeros(batch_size + (stack_size+1, hidden_size, num_layers), device=device)
    self.trees = torch.zeros(batch_size + (stack_size, input_size), device=device)

    if beam_size == 1:
      self.hiddens[:, 0] = initial_hidden[0].float()
      self.cells[:, 0] = initial_hidden[1].float()
    else:
      # Only fill zero-th beam position because we do not have other beam elems at beginning of search.
      self.hiddens[:, 0, 0] = initial_hidden[0].float()
      self.cells[:, 0, 0] = initial_hidden[1].float()

    self.nt_index = torch.zeros(batch_size + (stack_size,), dtype=torch.long, device=device)
    self.nt_ids = torch.zeros(batch_size + (stack_size,), dtype=torch.long, device=device)
    self.nt_index_pos = torch.tensor([-1], dtype=torch.long, device=device).expand(batch_size).clone() # default is -1 (0 means zero-dim exists)

  def hidden_head(self, offset = 0, batches = None):
    assert offset >= 0
    if batches is None:
      return self.hiddens[self.batch_index + (self.top_position.view(-1)-offset,)]
    else:
      return self.hiddens[batches + (self.top_position[batches]-offset,)]

  def cell_head(self, offset = 0, batches = None):
    assert offset >= 0
    if batches is None:
      return self.cells[self.batch_index + ((self.top_position.view(-1)-offset),)]
    else:
      return self.cells[batches + (self.top_position[batches]-offset,)]

  def do_shift(self, shift_batches, shifted_embs):
    self.trees[shift_batches + (self.top_position[shift_batches],)] = shifted_embs
    self.pointer[shift_batches] = self.pointer[shift_batches] + 1
    self.top_position[shift_batches] = self.top_position[shift_batches] + 1

  def do_nt(self, nt_batches, nt_embs, nt_ids):
    self.trees[nt_batches + (self.top_position[nt_batches],)] = nt_embs

    self.nt_index_pos[nt_batches] = self.nt_index_pos[nt_batches] + 1
    self.nt_ids[nt_batches + (self.nt_index_pos[nt_batches],)] = nt_ids
    self.top_position[nt_batches] = self.top_position[nt_batches] + 1
    self.nt_index[nt_batches + (self.nt_index_pos[nt_batches],)] = self.top_position[nt_batches]

  def do_reduce(self, reduce_batches, new_child):
    prev_nt_position = self.nt_index[reduce_batches + (self.nt_index_pos[reduce_batches],)]
    self.trees[reduce_batches + (prev_nt_position-1,)] = new_child.float()
    self.nt_index_pos[reduce_batches] = self.nt_index_pos[reduce_batches] - 1
    self.top_position[reduce_batches] = prev_nt_position

  def collect_reduced_children(self, reduce_batches):
    """

    :param reduce_batches: Tuple of idx tensors (output of non_zero()).
    """
    nt_index_pos = self.nt_index_pos[reduce_batches]
    prev_nt_position = self.nt_index[reduce_batches + (nt_index_pos,)]
    reduced_nt_ids = self.nt_ids[reduce_batches + (nt_index_pos,)]
    reduced_nts = self.trees[reduce_batches + (prev_nt_position-1,)]
    child_length = self.top_position[reduce_batches] - prev_nt_position
    max_ch_length = child_length.max()

    child_idx = prev_nt_position.unsqueeze(1) + torch.arange(max_ch_length, device=prev_nt_position.device)
    child_idx[child_idx >= self.stack_size] = self.stack_size - 1  # ceiled at maximum stack size (exceeding this may occur for some batches, but those should be ignored safely.)
    child_idx = child_idx.unsqueeze(-1).expand(-1, -1, self.trees.size(-1))  # (num_reduced_batch, max_num_child, input_dim)
    reduced_children = torch.gather(self.trees[reduce_batches], 1, child_idx)
    return reduced_children, child_length, reduced_nts, reduced_nt_ids

  def update_hidden(self, new_hidden, new_cell):
    pos = self.top_position.reshape(-1).clone()
    self.hiddens[self.batch_index + (pos,)] = new_hidden.float()
    self.cells[self.batch_index + (pos,)] = new_cell.float()

  def reset_stack(self):
    # may be useful for reusing stack.
    self.pointer[:] = 0
    self.nt_index_pos[:] = -1

    self.nopen_parens = [0] * batch_size
    self.ncons_nts = [0] * batch_size

  def sort_by(self, sort_idx):
    """

    :param sort_idx: (batch_size, beam_size) or (batch_size)
    """

    def sort_tensor(tensor):
      _idx = sort_idx
      for i in range(sort_idx.dim(), tensor.dim()):
        _idx = _idx.unsqueeze(-1)
      return torch.gather(tensor, sort_idx.dim()-1, _idx.expand(tensor.size()))

    self.pointer = sort_tensor(self.pointer)
    self.top_position = sort_tensor(self.top_position)
    self.hiddens = sort_tensor(self.hiddens)
    self.cells = sort_tensor(self.cells)
    self.trees = sort_tensor(self.trees)
    self.nt_index = sort_tensor(self.nt_index)
    self.nt_ids = sort_tensor(self.nt_ids)
    self.nt_index_pos = sort_tensor(self.nt_index_pos)

  def move_beams(self, self_move_idxs, other, move_idxs):
    self.pointer[self_move_idxs] = other.pointer[move_idxs]
    self.top_position[self_move_idxs] = other.top_position[move_idxs]
    self.hiddens[self_move_idxs] = other.hiddens[move_idxs]
    self.cells[self_move_idxs] = other.cells[move_idxs]
    self.trees[self_move_idxs] = other.trees[move_idxs]
    self.nt_index[self_move_idxs] = other.nt_index[move_idxs]
    self.nt_ids[self_move_idxs] = other.nt_ids[move_idxs]
    self.nt_index_pos[self_move_idxs] = other.nt_index_pos[move_idxs]


class BeamItems:
  def __init__(self, stack, max_actions = 500, beam_is_empty = False):
    self.batch_size = stack.batch_size
    self.beam_size = stack.beam_size
    self.stack = stack

    self.gen_ll = torch.tensor([-float('inf')], device=stack.hiddens.device).expand(
      self.batch_size, self.beam_size).clone()
    self.disc_ll = torch.tensor([-float('inf')], device=stack.hiddens.device).expand(
      self.batch_size, self.beam_size).clone()
    self.gen_ll[:, 0] = 0
    self.disc_ll[:, 0] = 0

    if beam_is_empty:
        # how many beam elements are active for each batch?
      self.beam_widths = self.gen_ll.new_zeros(self.batch_size, dtype=torch.long)
    else:
      self.beam_widths = self.gen_ll.new_ones(self.batch_size, dtype=torch.long)

    self.ncons_nts = self.beam_widths.new_zeros((self.batch_size, self.beam_size))
    self.nopen_parens = self.beam_widths.new_zeros((self.batch_size, self.beam_size))

    self.actions = self.beam_widths.new_full((self.batch_size, self.beam_size, max_actions), -1)
    self.actions_pos = self.beam_widths.new_zeros(self.batch_size, self.beam_size)

  def prev_actions(self):
    return self.actions.gather(2, self.actions_pos.unsqueeze(-1)).squeeze(-1)  # (batch_size, beam_size)

  def nbest_parses(self):
    widths = self.beam_widths.cpu().numpy()
    actions = self.actions.cpu().numpy()
    actions_pos = self.actions_pos.cpu().numpy()

    def tree_actions(batch, beam):
      return (actions[batch, beam, 1:actions_pos[batch, beam]+1], self.gen_ll[batch, beam].item())

    def batch_actions(batch):
      return [tree_actions(batch, i) for i in range(widths[batch])]

    return [batch_actions(b) for b in range(len(widths))]

  def shrink(self, size):
    outside_beam_idx = (torch.arange(self.beam_size, device=self.gen_ll.device).unsqueeze(0) >=
                        self.beam_widths.unsqueeze(1)).nonzero(as_tuple=True)
    self.gen_ll[outside_beam_idx] = -float('inf')
    self.gen_ll, sort_idx = torch.sort(self.gen_ll, descending=True)
    self.disc_ll = torch.gather(self.disc_ll, 1, sort_idx)
    self.stack.sort_by(sort_idx)
    self.beam_widths = torch.min(self.beam_widths, self.beam_widths.new_tensor([size]))
    self.ncons_nts = torch.gather(self.ncons_nts, 1, sort_idx)
    self.nopen_parens = torch.gather(self.nopen_parens, 1, sort_idx)

    self.actions = torch.gather(self.actions, 1, sort_idx.unsqueeze(-1).expand(self.actions.size()))
    self.actions_pos = torch.gather(self.actions_pos, 1, sort_idx)

  def active_idxs(self):
    """
    :return (batch_idxs, beam_idxs): All active idxs according to active beam sizes for each batch
                                     defined by self.beam_widths.
    """
    return (self.active_idx_mask() == 1).nonzero(as_tuple=True)

  def active_idx_mask(self):
    order = torch.arange(self.beam_size, device=self.beam_widths.device)
    return order < self.beam_widths.unsqueeze(1)

  def clear(self):
    self.beam_widths[:] = 0

  def move_items_from(self, other, move_idxs, new_gen_ll = None, new_disc_ll = None):
    """

    :param other: BeamItems
    :param move_idxs: A pair of index tensors (for batch_index and beam_index)
    :param new_gen_ll: If not None, replace gen_ll of the target positions with this vector.
    :param new_disc_ll: If not None, replace disc_ll of the target positions with this vector.
    """
    assert len(move_idxs) == 2  # hard-coded for beam search case.
    # This method internally presupposes that batch_index is sorted.
    assert torch.equal(move_idxs[0].sort()[0], move_idxs[0])
    move_batch_idxs, move_beam_idxs = move_idxs

    batch_numbers = bincount_and_supply(move_batch_idxs, self.batch_size)
    max_moved_beam_size = batch_numbers.max()
    new_beam_widths = self.beam_widths + batch_numbers  # (batch_size)
    if (new_beam_widths.max() > self.beam_size):
      # When exceeding max beam size.
      # This may happen when moving shifted elements to the word-finished BeamItems, maybe
      # especially when shift_size is larger.
      # We don't save this case and discard elements not fitted in self.beam_size.
      beam_idx_order = torch.arange(max_moved_beam_size, device=batch_numbers.device)
      sum_beam_idx_order =  self.beam_widths.unsqueeze(1) + beam_idx_order
      move_idx_mask = sum_beam_idx_order < self.beam_size
      move_idx_mask = move_idx_mask.view(-1)[
        (beam_idx_order < batch_numbers.unsqueeze(1)).view(-1)]
      move_idxs = (move_idxs[0][move_idx_mask], move_idxs[1][move_idx_mask])
      move_batch_idxs, move_beam_idxs = move_idxs
      if new_gen_ll is not None:
        new_gen_ll = new_gen_ll[move_idx_mask]
      if new_disc_ll is not None:
        new_disc_ll = new_disc_ll[move_idx_mask]
      batch_numbers = bincount_and_supply(move_batch_idxs, self.batch_size)
      max_moved_beam_size = batch_numbers.max()
      new_beam_widths = self.beam_widths + batch_numbers  # (batch_size)

    self_move_beam_idxs = self.beam_widths.unsqueeze(1) + torch.arange(
      max_moved_beam_size, device=batch_numbers.device)
    self_beam_idx_mask = self_move_beam_idxs < new_beam_widths.unsqueeze(1)
    self_move_beam_idxs = self_move_beam_idxs.view(-1)[self_beam_idx_mask.view(-1).nonzero(as_tuple=True)]
    assert self_move_beam_idxs.size() == move_beam_idxs.size()

    self_move_idxs = (move_batch_idxs, self_move_beam_idxs)
    self.beam_widths = new_beam_widths
    self.gen_ll[self_move_idxs] = new_gen_ll if new_gen_ll is not None else other.gen_ll[move_idxs]
    self.disc_ll[self_move_idxs] = new_disc_ll if new_disc_ll is not None else other.disc_ll[move_idxs]
    self._do_move_elements(other, self_move_idxs, move_idxs, new_gen_ll, new_disc_ll)

    return self_move_idxs

  def reconstruct(self, target_idxs):
    """

    Intuitively perform beam[:] = beam[target_idxs]. target_idxs contains duplicates so this would
    copy some elements across different idxs. A core function in beam search.
    """
    assert self.beam_widths.sum() > 0

    assert len(target_idxs) == 2  # hard-coded for beam search case.
    move_batch_idxs, move_beam_idxs = target_idxs
    self.beam_widths = bincount_and_supply(move_batch_idxs, self.batch_size)
    assert self.beam_widths.max() <= self.beam_size

    self_move_beam_idxs = (torch.arange(self.beam_widths.max(), device=target_idxs[0].device)
                           .unsqueeze(0).repeat(self.beam_widths.size(0), 1))
    self_beam_idx_mask = self_move_beam_idxs < self.beam_widths.unsqueeze(1)
    self_move_beam_idxs = self_move_beam_idxs.view(-1)[
      self_beam_idx_mask.view(-1).nonzero(as_tuple=True)]
    assert self_move_beam_idxs.size() == move_beam_idxs.size()

    self_move_idxs = (move_batch_idxs, self_move_beam_idxs)
    self._do_move_elements(self, self_move_idxs, target_idxs)

    return self_move_idxs

  def marginal_probs(self):
    active_idx_mask = self.active_idx_mask()
    self.gen_ll[active_idx_mask != 1] = -float('inf')
    return torch.logsumexp(self.gen_ll, 1)

  def _do_move_elements(self, source, self_idxs, source_idxs, new_gen_ll = None, new_disc_ll = None):
    self.gen_ll[self_idxs] = new_gen_ll if new_gen_ll is not None else source.gen_ll[source_idxs]
    self.disc_ll[self_idxs] = new_disc_ll if new_disc_ll is not None else source.disc_ll[source_idxs]

    self.ncons_nts[self_idxs] = source.ncons_nts[source_idxs]
    self.nopen_parens[self_idxs] = source.nopen_parens[source_idxs]
    self.stack.move_beams(self_idxs, source.stack, source_idxs)

    self.actions[self_idxs] = source.actions[source_idxs]
    self.actions_pos[self_idxs] = source.actions_pos[source_idxs]

  def do_action(self, actions, action_dict):
    active_idxs = self.active_idxs()
    self.actions_pos[active_idxs] += 1
    self.actions[active_idxs + (self.actions_pos[active_idxs],)] = actions[active_idxs]

    self._update_nt_counts(actions, action_dict)

  def _update_nt_counts(self, actions, action_dict):
    shift_idxs = (actions == action_dict.a2i['SHIFT']).nonzero(as_tuple=True)
    nt_idxs = (actions >= action_dict.nt_begin_id()).nonzero(as_tuple=True)
    reduce_idxs = (actions == action_dict.a2i['REDUCE']).nonzero(as_tuple=True)

    self.ncons_nts[shift_idxs] = 0
    self.nopen_parens[nt_idxs] += 1
    self.ncons_nts[nt_idxs] += 1
    self.nopen_parens[reduce_idxs] -= 1
    self.ncons_nts[reduce_idxs] = 0

class RNNGCell(nn.Module):
  """
  RNNGCell receives next action and input word embedding, do action, and returns next updated hidden states.
  """
  def __init__(self,
               input_size,
               hidden_size,
               num_layers,
               dropout,
               action_dict,
               attention_composition):
    super(RNNGCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)

    self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts(), input_size), self.dropout)
    self.stack_rnn = MultiLayerLSTMCell(input_size, hidden_size, num_layers, dropout=dropout)
    self.output = nn.Sequential(self.dropout, nn.Linear(hidden_size, input_size), nn.ReLU())
    self.composition = (AttentionComposition(input_size, dropout, action_dict.num_nts())
                        if attention_composition else
                        LSTMComposition(input_size, dropout))

    self.initial_emb = nn.Sequential(nn.Embedding(1, input_size), self.dropout)

    self.action_dict = action_dict

  def get_initial_hidden(self, x):
    iemb = self.initial_emb(x.new_zeros(x.size(0), dtype=torch.long))
    return self.stack_rnn(iemb, None)

  def forward(self, word_vecs, actions, stack):
    """
    Similar to update_stack_rnn.

    :param word_vecs: (batch_size, sent_len, input_size)
    :param actions: (batch_size, 1)
    """

    reduce_batches = (actions == self.action_dict.a2i['REDUCE']).nonzero(as_tuple=True)
    nt_batches = (actions >= self.action_dict.nt_begin_id()).nonzero(as_tuple=True)
    shift_batches = (actions == self.action_dict.a2i['SHIFT']).nonzero(as_tuple=True)

    new_input = word_vecs.new_zeros(stack.hiddens.size()[:-3] + (self.input_size,))

    # First fill in trees. Then, gather those added elements in a column, which become
    # the input to stack_rnn.
    if shift_batches[0].size(0) > 0:
      shift_idx = stack.pointer[shift_batches].view(-1, 1, 1).expand(-1, 1, word_vecs.size(-1))
      shifted_embs = torch.gather(word_vecs[shift_batches[0]], 1, shift_idx).squeeze(1)
      stack.do_shift(shift_batches, shifted_embs)
      new_input[shift_batches] = shifted_embs

    if nt_batches[0].size(0) > 0:
      nt_ids = (actions[nt_batches] - self.action_dict.nt_begin_id())
      nt_embs = self.nt_emb(nt_ids)
      stack.do_nt(nt_batches, nt_embs, nt_ids)
      new_input[nt_batches] = nt_embs

    if reduce_batches[0].size(0) > 0:
      children, ch_lengths, reduced_nt, reduced_nt_ids = stack.collect_reduced_children(reduce_batches)
      if isinstance(self.composition, AttentionComposition):
        hidden_head = stack.hidden_head(batches=reduce_batches)[:, :, -1]
        stack_h = self.output(hidden_head)
      else:
        stack_h = None
      new_child, _, _ = self.composition(children, ch_lengths, reduced_nt, reduced_nt_ids, stack_h)
      stack.do_reduce(reduce_batches, new_child)
      new_input[reduce_batches] = new_child.float()

    # Input for rnn should be (beam_size, input_size). During beam search, new_input has different size.
    new_hidden, new_cell = self.stack_rnn(new_input.view(-1, self.input_size),
                                          (stack.hidden_head(1), stack.cell_head(1)))

    stack.update_hidden(new_hidden, new_cell)

    return stack.hidden_head()[..., -1]


class FixedStackRNNG(nn.Module):
  def __init__(self, action_dict,
               vocab = 100,
               padding_idx = 0,
               w_dim = 20,
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               attention_composition=False,
               max_open_nts = 100,
               max_cons_nts = 8,
  ):
    super(FixedStackRNNG, self).__init__()
    self.action_dict = action_dict
    self.padding_idx = padding_idx
    self.action_criterion = nn.CrossEntropyLoss(reduction='none',
                                                ignore_index=action_dict.padding_idx)
    self.word_criterion = nn.CrossEntropyLoss(reduction='none',
                                              ignore_index=padding_idx)

    self.dropout = nn.Dropout(dropout)
    self.emb = nn.Sequential(
      nn.Embedding(vocab, w_dim, padding_idx=padding_idx), self.dropout)

    self.rnng = RNNGCell(w_dim, h_dim, num_layers, dropout, self.action_dict, attention_composition)

    self.vocab_mlp = nn.Linear(w_dim, vocab)
    self.num_layers = num_layers
    self.num_actions = action_dict.num_actions()  # num_labels + 2
    self.action_mlp = nn.Linear(w_dim, self.num_actions)
    self.input_size = w_dim
    self.hidden_size = h_dim
    self.vocab_mlp.weight = self.emb[0].weight

    self.max_open_nts = max_open_nts
    self.max_cons_nts = max_cons_nts

  def forward(self, x, actions, initial_stack = None):
    assert isinstance(x, torch.Tensor)
    assert isinstance(actions, torch.Tensor)

    stack = self.build_stack(x)
    word_vecs = self.emb(x)
    action_contexts = self.unroll_states(stack, word_vecs, actions)

    a_loss, _ = self.action_loss(actions, self.action_dict, action_contexts)
    w_loss, _ = self.word_loss(x, actions, self.action_dict, action_contexts)
    loss = (a_loss.sum() + w_loss.sum())
    return loss, a_loss, w_loss, stack

  def unroll_states(self, stack, word_vecs, actions):
    hs = word_vecs.new_zeros(actions.size(1), word_vecs.size(0), self.hidden_size)
    hs[0] = stack.hidden_head()[:, :, -1]
    for step in range(actions.size(1)-1):
      h = self.rnng(word_vecs, actions[:, step], stack)  # (batch_size, input_size)
      hs[step+1] = h
    hs = self.rnng.output(hs.transpose(1, 0).contiguous())  # (batch_size, action_len, input_size)
    return hs

  def build_stack(self, x):
    stack_size = max(150, x.size(1) + 100)
    initial_hidden = self.rnng.get_initial_hidden(x)
    return FixedStack(initial_hidden, stack_size, self.input_size)

  def action_loss(self, actions, action_dict, hiddens):
    assert hiddens.size()[:2] == actions.size()
    actions = actions.view(-1)
    hiddens = hiddens.view(actions.size(0), -1)

    action_mask = actions != action_dict.padding_idx
    idx = action_mask.nonzero().squeeze(1)
    actions = actions[idx]
    hiddens = hiddens[idx]

    logit = self.action_mlp(hiddens)
    loss = self.action_criterion(logit, actions)
    return loss, logit

  def word_loss(self, x, actions, action_dict, hiddens):
    assert hiddens.size()[:2] == actions.size()
    actions = actions.view(-1)
    hiddens = hiddens.view(actions.size(0), -1)

    action_mask = actions == action_dict.a2i['SHIFT']
    idx = action_mask.nonzero().squeeze(1)
    hiddens = hiddens[idx]

    x = x.view(-1)
    assert x.size(0) == hiddens.size(0)
    logit = self.vocab_mlp(hiddens)
    loss = self.word_criterion(logit, x)
    return loss, logit

  def word_sync_beam_search(self, x, beam_size, word_beam_size = 0, shift_size = 0,
                            delay_word_ll = False, return_beam_history = False):
    self.eval()
    if (hasattr(self.rnng.composition, 'batch_index') and
        self.rnng.composition.batch_index.size(0) < x.size(0)*beam_size):
      # The maximum number may be set by assuming training setting only.
      # Here we reset to the maximum number by beam search.
      self.rnng.composition.batch_index = torch.arange(0, x.size(0)*beam_size, device=x.device)

    if word_beam_size <= 0:
      word_beam_size = beam_size

    beam, word_completed_beam = self.build_beam_items(x, beam_size, shift_size)
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      forced_completions = x.new_zeros(x.size(0), dtype=torch.long)
      bucket_i = 0

      def word_finished_batches():
        return ((beam.beam_widths == 0) +  # Empty beam means no action remains (finished).
                #(word_completed_beam.beam_widths >= (word_completed_beam.beam_size - 1)) +
                (word_completed_beam.beam_widths >= word_completed_beam.beam_size) +
                ((word_completed_beam.beam_widths - forced_completions) >= beam_size))

      finished_batches = word_finished_batches()

      while not finished_batches.all():
        added_forced_completions = self.beam_step(
          x, word_vecs, pointer, beam, word_completed_beam, shift_size)
        forced_completions += added_forced_completions
        finished_batches = word_finished_batches()
        beam.beam_widths[finished_batches.nonzero(as_tuple=True)] = 0  # inactive word-finished batches.
        bucket_i += 1

      self.finalize_word_completed_beam(
        x, word_vecs, pointer, beam, word_completed_beam, word_beam_size)

      marginal = beam.marginal_probs()
      for b, s in enumerate(marginal.cpu().detach().numpy()):
        word_marginal_ll[b].append(s)

    parses = beam.nbest_parses()
    surprisals = [[] for _ in range(x.size(0))]
    for b in range(x.size(0)):
      for i in range(0, len(word_marginal_ll[b])-1):
        surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))

    ret = (parses, surprisals)
    return ret

  def beam_step(self, x, word_vecs, pointer, beam, word_completed_beam, shift_size):
    beam_size = beam.beam_size
    successors, word_completed_successors, added_forced_completions \
      = self.get_successors(x, pointer, beam, beam_size, shift_size)

    # tuple of ((batch_idxs, beam_idxs), next_actions, total_scores)
    assert len(successors) == len(word_completed_successors) == 3

    if word_completed_successors[0][0].size(0) > 0:
      comp_idxs = tuple(word_completed_successors[0][:2])
      # Add elements to word_completed_beam
      # This assumes that returned scores are total scores rather than the current action scores.
      moved_idxs = word_completed_beam.move_items_from(
        beam, comp_idxs, new_gen_ll=word_completed_successors[2])

    new_beam_idxs = beam.reconstruct(successors[0][:2])
    beam.gen_ll[new_beam_idxs] = successors[2]
    actions = successors[1].new_full((x.size(0), beam_size), self.action_dict.padding_idx)
    actions[new_beam_idxs] = successors[1]
    self.rnng(word_vecs, actions, beam.stack)
    beam.do_action(actions, self.action_dict)

    return added_forced_completions

  def finalize_word_completed_beam(
      self, x, word_vecs, pointer, beam, word_completed_beam, word_beam_size):
    beam_size = word_completed_beam.beam_size
    word_completed_beam.shrink(word_beam_size)
    word_end_actions = x.new_full((x.size(0), beam_size), self.action_dict.padding_idx)
    active_idx = word_completed_beam.active_idxs()
    if pointer < x.size(1):  # do shift
      word_end_actions[active_idx] = self.action_dict.a2i['SHIFT']
    else:
      word_end_actions[active_idx] = self.action_dict.finish_action()
    self.rnng(word_vecs, word_end_actions, word_completed_beam.stack)
    word_completed_beam.do_action(word_end_actions, self.action_dict)

    beam.clear()
    beam.move_items_from(word_completed_beam, active_idx)
    word_completed_beam.clear()

  def build_beam_items(self, x, beam_size, shift_size):
    #stack_size = max(100, x.size(1) + 20)
    stack_size = min(int(x.size(1)*2.5), 104)
    stack_size = math.ceil(stack_size / 8) * 8
    #stack_size = max(10, stack_size)
    initial_hidden = self.rnng.get_initial_hidden(x)
    stack_for_unfinished = FixedStack(initial_hidden, stack_size, self.input_size, beam_size)
    # The rationale behind (+shift_size*5) for beam size for finished BeamItems is
    # that # steps between words would probably be ~5 in most cases. Forcing to save shifts
    # after this many steps seems to be unnecessary.
    stack_for_word_finished = FixedStack(initial_hidden, stack_size, self.input_size,
                                         min(beam_size * 2, beam_size + shift_size*5))
    max_actions = max(100, x.size(1) * 5)
    return (BeamItems(stack_for_unfinished, max_actions, False),
            BeamItems(stack_for_word_finished, max_actions, True))

  def variable_beam_search(self, x, K, original_reweight=False):
    self.eval()
    if (hasattr(self.rnng.composition, 'batch_index') and
        self.rnng.composition.batch_index.size(0) < x.size(0)*10000):
      # The maximum number may be set by assuming training setting only.
      # Here we reset to the maximum number by beam search.
      self.rnng.composition.batch_index = torch.arange(0, x.size(0)*10000, device=x.device)


    beam = self.initial_particle_beam(x, K)
    word_completed = [[] for _ in range(x.size(0))]
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      bucket_i = 0
      while not all(len(batch_beam) == 0 for batch_beam in beam):
        new_beam = self.get_successors_by_particle_filter(x, pointer, beam)
        all_items, batch_idx = self._flatten_items(new_beam)
        self.update_stack_rnn_beam(all_items, word_vecs, batch_idx, pointer)
        beam_lengths = [len(batch_beam) for batch_beam in new_beam]
        self.update_beam_and_word_completed(
          beam, word_completed, all_items, beam_lengths, pointer == x.size(1))
        bucket_i += 1

      for b in range(len(beam)):
        beam[b] = self.reweight_and_filter_particles(
          word_completed[b], K, original_reweight=original_reweight)
        word_completed[b] = []

      marginal = self._get_marginal_ll(beam)
      for b, s in enumerate(marginal):
        word_marginal_ll[b].append(s)

    parses = [sorted([(item.parse_actions(), item.score) for item in batch_beam],
                     key=lambda x:x[1], reverse=True)
              for batch_beam in beam]
    surprisals = [[] for _ in range(len(beam))]
    for b in range(len(beam)):
      for i in range(0, len(word_marginal_ll[b])-1):
        surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))
    return parses, surprisals

  def initial_beam(self, x):
    states = self.initial_states(x)
    return [[BeamItem.from_initial_state(state)] for state in states]

  def initial_states(self, x):
    initial_hidden = self.rnng.get_initial_hidden(x)  # [(batch_size, hidden_size, layer), (batch_size, hidden_size, layer)]
    return [TopDownState.from_initial_stack((initial_hidden[0][b], initial_hidden[1][b]))
            for b in range(x.size(0))]

  def initial_particle_beam(self, x, K):
    states = self.initial_states(x)
    return [[ParticleBeamItem.from_initial_state(state, K)] for state in states]

  def get_successors(self, x, pointer, beam, beam_size, shift_size):
    if pointer < x.size(1):
      next_x = x[:, pointer]
    else:
      next_x = None

    invalid_action_mask = self.invalid_action_mask(beam, x.size(1))  # (total beam size, n_actions)

    log_probs = self.action_log_probs(beam.stack, invalid_action_mask, next_x)  # (batch, beam, n_actions)
    # scores for inactive beam items (outside active_idx) are -inf on log_probs so we need
    # not worry about values in gen_ll outside active_idx.
    log_probs += beam.gen_ll.unsqueeze(-1)

    return self.scores_to_successors(x, pointer, beam, log_probs, beam_size, shift_size)

  def get_successors_by_particle_filter(self, x, pointer, beam):
    all_items, _ = self._flatten_items(beam)
    beam_lengths = [len(batch_beam) for batch_beam in beam]
    states = [item.state for item in all_items]

    if pointer < x.size(1):
      next_x = [x[b, pointer].expand(beam_lengths[b]) for b in range(x.size(0))]
      next_x = torch.cat(next_x)  # (total beam size)
    else:
      next_x = None

    action_mask = self.valid_action_mask(all_items, x.size(1))  # (total beam size, n_actions)
    log_probs, disc_log_probs = self.action_log_probs(states, action_mask, next_x, return_disc_probs=True)
    new_K = (disc_log_probs.exp() * log_probs.new_tensor(
      [item.particle_path.K for item in all_items]).unsqueeze(1)).round_()
    new_K = new_K.view(-1)
    mask = new_K > 0.0  # (total beam size * n_actions)
    # action id can be obtained by (idx % n_actions)

    offset = 0
    successors = []
    num_actions = log_probs.size(1)  # (total_beam_size*n_actions, 1)
    log_probs = log_probs.view(-1)
    disc_log_probs = disc_log_probs.view(-1)
    for batch_i, beam_length in enumerate(beam_lengths):
      b, e = offset*num_actions, (offset+beam_length)*num_actions
      batch_active_idx = mask[b:e].nonzero().squeeze(1)
      beam_id = (batch_active_idx // num_actions).cpu().numpy()
      action_id = (batch_active_idx % num_actions).cpu().numpy()
      active_idx = batch_active_idx + offset*num_actions
      batch_K = new_K[active_idx].cpu().numpy()
      batch_log_probs = log_probs[active_idx].cpu().numpy()
      batch_disc_log_probs = disc_log_probs[active_idx].cpu().numpy()

      successors.append([beam[batch_i][beam_id[i]].next_incomplete_item(
        action_id[i], batch_K[i], batch_log_probs[i], batch_disc_log_probs[i])
                         for i in range(len(beam_id))])
      offset += beam_length

    return successors

  def reweight_and_filter_particles(self, batch_beam, K, upper_lex_size=1000,
                                    original_reweight=False):
    if original_reweight:
      scores = torch.tensor([[b.particle_path.K,
                              b.particle_path.gen_ll,
                              b.particle_path.disc_ll] for b in batch_beam])
      scores = torch.t(scores)
      log_weights = scores[0].log() + scores[1] - scores[2]
    else:
      log_weights = torch.tensor([b.particle_path.gen_ll for b in batch_beam])

    denom = torch.logsumexp(log_weights, 0)
    unround_Ks = (log_weights - denom).exp() * K
    new_Ks = unround_Ks.round()

    active = (new_Ks > 0).nonzero().squeeze(1)
    if active.size(0) > upper_lex_size:
      active_Ks = unround_Ks[active]
      _, sort_idx = torch.sort(active_Ks, descending=True)
      active = active[sort_idx[:upper_lex_size]].cpu().numpy()
    new_Ks = new_Ks.cpu().numpy()
    new_beam = []
    for i in active:
      b = batch_beam[i]
      b.particle_path.reweight(new_Ks[i])
      new_beam.append(b)

    return new_beam

  def action_log_probs(self, stack, invalid_action_mask, next_x = None, return_disc_probs = False):
    """

    :param stack: FixedStack
    :param invalid_action_mask: (batch_size, beam_size, num_actions)  (inactive beams are entirely masked.)
    :param next_x: (batch_size) to be shifted token ids
    """
    hiddens = self.rnng.output(stack.hidden_head()[:, :, -1])  # (beam*batch, hidden_size)
    action_logit = self.action_mlp(hiddens).view(invalid_action_mask.size())  # (beam, batch, num_actions)
    action_logit[invalid_action_mask] = -float('inf')

    log_probs = F.log_softmax(action_logit, -1)  # (batch_size, beam_size, num_actions)
    log_probs[torch.isnan(log_probs)] = -float('inf')
    if return_disc_probs:
      disc_log_probs = log_probs.clone()

    if next_x is not None:  # shift is valid for next action
      word_logit = self.vocab_mlp(hiddens)  # (batch*beam, vocab_size)
      shift_idx = self.action_dict.a2i['SHIFT']
      next_x = next_x.unsqueeze(1).expand(-1, log_probs.size(1)).clone().view(-1)  # (batch*beam)
      word_log_probs = self.word_criterion(word_logit, next_x) * -1.0  # (batch_size*beam_size, vocab_size)
      word_log_probs = word_log_probs.view(log_probs.size(0), log_probs.size(1))
      log_probs[:, :, shift_idx] += word_log_probs

    if return_disc_probs:
      return (log_probs, disc_log_probs)
    else:
      return log_probs

  def invalid_action_mask(self, beam, sent_len):
    """Return a tensor where mask[i,j,k]=True means action k is not allowed for beam (i,j).
    """
    action_order = torch.arange(self.num_actions, device=beam.nopen_parens.device)

    # reduce_mask[i,j,k]=True means k is a not allowed reduce action for (i,j).
    reduce_mask = (action_order == self.action_dict.a2i['REDUCE']).view(1, 1, -1)
    reduce_mask = ((((beam.nopen_parens == 1) * (beam.stack.pointer < sent_len)) +
                    # prev is nt => cannot reduce immediately after nt
                    (beam.prev_actions() >= self.action_dict.nt_begin_id()) +
                    (beam.stack.top_position < 2)).unsqueeze(-1) *
                   reduce_mask)

    # nt_mask[i,j,k]=True means k is a not allowed nt action for (i,j).
    nt_mask = (action_order >= self.action_dict.nt_begin_id()).view(1, 1, -1)
    nt_mask = (((beam.nopen_parens >= self.max_open_nts) +
                (beam.ncons_nts >= self.max_cons_nts) +
                # Check the storage of beam.actions, which is bounded beforehand.
                (beam.actions.size(2) - beam.actions_pos < (
                  sent_len - beam.stack.pointer + beam.nopen_parens + 1)) +
                # Check the storage of fixed stack size (we need minimally two additional
                # elements to process arbitrary future structure).
                (beam.stack.top_position >= beam.stack.stack_size-2)).unsqueeze(-1) *
               nt_mask)

    shift_mask = (action_order == self.action_dict.a2i['SHIFT']).view(1, 1, -1)
    shift_mask = (beam.stack.top_position >= beam.stack.stack_size-1).unsqueeze(-1) * shift_mask

    # all actions other than nt are invalid;
    # except_nt_mask[i,j,k]=True means k (not nt) is not allowed for (i,j).
    except_nt_mask = (action_order < self.action_dict.nt_begin_id()).view(1, 1, -1)
    except_nt_mask = (beam.nopen_parens == 0).unsqueeze(-1) * except_nt_mask

    except_reduce_mask = (action_order != self.action_dict.a2i['REDUCE']).view(1, 1, -1)
    except_reduce_mask = (beam.stack.pointer == sent_len).unsqueeze(-1) * except_reduce_mask

    pad_mask = (action_order == self.action_dict.padding_idx).view(1, 1, -1)
    finished_mask = ((beam.stack.pointer == sent_len) * (beam.nopen_parens == 0)).unsqueeze(-1)
    beam_width_mask = (torch.arange(beam.beam_size, device=reduce_mask.device).unsqueeze(0) >=
                       beam.beam_widths.unsqueeze(1)).unsqueeze(-1)

    return (reduce_mask + nt_mask + shift_mask + except_nt_mask + except_reduce_mask +
            pad_mask + finished_mask + beam_width_mask)

  def stack_top_h(self, states):
    return torch.stack([state.hiddens[-1][:, -1] for state in states], dim=0)

  def scores_to_successors(self, x, pointer, beam, total_scores, beam_size, shift_size):
    assert (total_scores.isnan() != 1).all()
    num_actions = total_scores.size(2)
    total_scores = total_scores.view(total_scores.size(0), -1)
    sorted_scores, sort_idx = torch.sort(total_scores, descending=True)

    beam_id = sort_idx // num_actions
    action_id = sort_idx % num_actions

    valid_action_mask = sorted_scores != -float('inf')

    if pointer < x.size(1):  #
      end_action_mask = action_id == self.action_dict.a2i['SHIFT']
    else:
      # top_down specific masking; should be separated in a function and adjusted for in_order.
      pre_final_mask = beam.nopen_parens.gather(1, beam_id) == 1
      end_action_mask = action_id == self.action_dict.finish_action()
      end_action_mask = end_action_mask * pre_final_mask

    end_action_mask = valid_action_mask * end_action_mask
    no_end_action_mask = valid_action_mask * (end_action_mask != 1)

    within_beam_end_action_idx = end_action_mask[:, :beam_size].nonzero(as_tuple=True)
    within_beam_num_end_actions = bincount_and_supply(
      within_beam_end_action_idx[0], x.size(0))  # (batch_size)
    # max num of forcefully shifted actions (actual number is upperbounded by active actions).
    num_to_be_forced_completions = torch.maximum(
      torch.tensor([0], device=x.device).expand(x.size(0)),
      shift_size - within_beam_num_end_actions)  # (batch_size)

    outside_end_action_mask = end_action_mask[:, beam_size:]
    outside_end_action_idx = outside_end_action_mask.nonzero(as_tuple=True)
    # outside_end_action_idx[0] may be [0, 0, 0, 1, 2, 2]
    # num_forced_completions might be [0, 1, 0];
    # pick up only 4th element, i.e., make a mask [F, F, F, T, F, F]
    #
    # strategy:
    #  outside_num_end_actions: [3, 1, 2]
    #  order: [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    #  size_recover_mask: [[T, T, T], [T, F, F], [T, T, F]]
    #  forced_completion_mask: [[F, F, F], [T, F, F], [F, F, F]]
    #  filter_mask: [F, F, F, T, F, F]
    outside_num_end_actions = bincount_and_supply(outside_end_action_idx[0], x.size(0))
    order = torch.arange(outside_num_end_actions.max(), device=x.device)
    size_recover_mask = order < outside_num_end_actions.unsqueeze(1)
    forced_completion_mask = order < num_to_be_forced_completions.unsqueeze(1)
    filter_mask = forced_completion_mask.view(-1)[size_recover_mask.view(-1)]
    outside_end_action_idx = (outside_end_action_idx[0][filter_mask],
                              outside_end_action_idx[1][filter_mask])

    outside_end_action_mask[:] = 0
    outside_end_action_mask[outside_end_action_idx] = 1

    num_forced_completions = bincount_and_supply(outside_end_action_idx[0], x.size(0))

    end_successor_idx = torch.cat(
      [end_action_mask[:, :beam_size], outside_end_action_mask], dim=1).nonzero(as_tuple=True)
    no_end_successor_idx = no_end_action_mask[:, :beam_size].nonzero(as_tuple=True)

    def successor_idx_to_successors(successor_idx):
      next_beam_ids = beam_id[successor_idx]
      next_action_ids = action_id[successor_idx]
      next_scores = sorted_scores[successor_idx]
      return (successor_idx[0], next_beam_ids), next_action_ids, next_scores

    return (successor_idx_to_successors(no_end_successor_idx),
            successor_idx_to_successors(end_successor_idx),
            num_forced_completions)


class TopDownState:
  """
  Previously this class is used both for training and inference.
  Now it is only used for beam search.

  TODO: work with FixedStack for beam search as well.
  """
  def __init__(self,
               pointer = 0,
               hiddens = None,
               cells = None,
               stack_trees = None,
               nopen_parens = 0,
               ncons_nts = 0,
               nt_index = None,
               nt_ids = None):
    self.pointer = pointer
    self.hiddens = hiddens or []  # each with (hidden_size, num_layers)
    self.cells = cells or []
    self.stack_trees = stack_trees or []
    self.nopen_parens = nopen_parens
    self.ncons_nts = ncons_nts
    self.nt_index = nt_index or []
    self.nt_ids = nt_ids or []

  def stack_str(self, action_dict):
    stack_str = ''
    stack_str = ['<dummy>'] + ['C' for _ in range(len(self.hiddens)-1)]
    for i, nt in zip(self.nt_index, self.nt_ids):
      stack_str[i] = '(' + action_dict.nonterminals[nt]
    assert self.nopen_parens == len(self.nt_ids) == len(self.nt_index)

    stack_str = ' '.join(stack_str)
    return '{{[ {} ], pointer={}}}'.format(stack_str, self.pointer)

  @classmethod
  def from_initial_stack(cls, initial_stack_elem):
    return cls(hiddens=[initial_stack_elem[0]], cells=[initial_stack_elem[1]])

  def can_finish_by_reduce(self):
    return self.nopen_parens == 1

  def finished(self):
    return self.pointer > 0 and self.nopen_parens == 0

  def copy(self):
    return TopDownState(self.pointer, self.hiddens[:], self.cells[:], self.stack_trees[:],
                        self.nopen_parens, self.ncons_nts, self.nt_index[:], self.nt_ids[:])

  def do_action(self, a, action_dict):
    if action_dict.is_shift(a):
      self.pointer += 1
      self.ncons_nts = 0
    elif action_dict.is_nt(a):
      nt_id = action_dict.nt_id(a)
      self.nopen_parens += 1
      self.ncons_nts += 1
      self.nt_index.append(len(self.hiddens) - 1)
      self.nt_ids.append(nt_id)
    elif action_dict.is_reduce(a):
      self.nopen_parens -= 1
      self.ncons_nts = 0

  def reduce_stack(self):
    open_idx = self.nt_index.pop()
    nt_id = self.nt_ids.pop()
    self.hiddens = self.hiddens[:open_idx]
    self.cells = self.cells[:open_idx]
    reduce_trees = self.stack_trees[open_idx-1:]
    self.stack_trees = self.stack_trees[:open_idx-1]
    return reduce_trees[0], reduce_trees[1:], nt_id

  def update_stack(self, new_stack_top, new_tree_elem, action = 0, action_dict = None):
    self.hiddens.append(new_stack_top[0])
    self.cells.append(new_stack_top[1])
    self.stack_trees.append(new_tree_elem)

class ActionPath:
  def __init__(self, prev=None, action=0, score=0.0, local_word_ll=0.0):
    self.prev = prev
    self.action = action
    self.score = score
    self.local_word_ll = local_word_ll

  def add_action(self, action, score, local_word_ll=0.0):
    return ActionPath(self, action, score, local_word_ll)

  def foreach(self, f):
    f(self)
    if self.prev is not None:
      self.prev.foreach(f)

  def incorporate_word_ll(self):
    self.local_word_ll < 0
    self.score += self.local_word_ll

class BeamItem:
  def __init__(self, state, action_path):
    self.state = state
    self.action_path = action_path
    self.action = self.action_path.action

  @property
  def score(self):
    return self.action_path.score

  def do_action(self, action_dict):
    self.state.do_action(self.action, action_dict)

  @staticmethod
  def from_initial_state(initial_state):
    path = ActionPath()  # initial action is 0 (pad)
    return BeamItem(initial_state, path)

  def parse_actions(self):
    actions = []
    def add_action(path):
      actions.append(path.action)
    self.action_path.foreach(add_action)
    assert actions[-1] == 0  # last (= initial after revsed) is pad (dummy)
    return list(reversed(actions[:-1]))

  def next_incomplete_item(self, action, score, local_word_ll=0.0):
    state = self.state.copy()
    path = self.action_path.add_action(action, score, local_word_ll)
    return BeamItem(state, path)

  def dump(self, action_dict, sent):
    actions = self.parse_actions()
    stack_str = action_dict.build_tree_str(
      actions, sent.orig_tokens, ["" for _ in range(len(sent.orig_tokens))])
    return "[{}]; {:.2f}; a={}, ".format(stack_str, self.score, self.action)

class ParticlePath:
  def __init__(self, K, prev=None, action=0, gen_ll=0.0, disc_ll=0.0):
    self.K = K
    self.prev = prev
    self.action = action
    self.gen_ll = gen_ll
    self.disc_ll = disc_ll

  def add_action(self, action, K, gen_local_score, disc_local_score):
    return ParticlePath(K, self, action, self.gen_ll+gen_local_score, self.disc_ll+disc_local_score)

  def reweight(self, new_K):
    self.K = new_K

  def foreach(self, f):
    f(self)
    if self.prev is not None:
      self.prev.foreach(f)

class ParticleBeamItem:
  def __init__(self, state, particle_path):
    self.state = state
    self.particle_path = particle_path
    self.action = self.particle_path.action

  @property
  def score(self):
    return self.particle_path.gen_ll

  @staticmethod
  def from_initial_state(initial_state, K):
    path = ParticlePath(K)  # initial action is 0 (pad)
    return ParticleBeamItem(initial_state, path)

  def parse_actions(self):
    actions = []
    def add_action(path):
      actions.append(path.action)
    self.particle_path.foreach(add_action)
    assert actions[-1] == 0  # last (= initial after revsed) is pad (dummy)
    return list(reversed(actions[:-1]))

  def next_incomplete_item(self, action, K, log_prob, disc_log_prob):
    state = self.state.copy()
    path = self.particle_path.add_action(action, K, log_prob, disc_log_prob)
    return ParticleBeamItem(state, path)
