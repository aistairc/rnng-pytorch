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
    self.dropout_layer = nn.Dropout(dropout, inplace=True)

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
    self.output = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(dim*2, dim), nn.ReLU())

    self.batch_index = torch.arange(0, 10000, dtype=torch.long)  # cache with sufficient number.

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
    self.dropout = nn.Dropout(dropout, inplace=True)

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


class ExpandableStorage:
  def __init__(self):
    self.attrs = []

  def expand_at_dim(self, target_dim, new_size):
    def same_dim_except_target(orig_size):
      if isinstance(orig_size, tuple):
        orig_size = list(orig_size)
      size_diff = new_size - orig_size[target_dim]
      orig_size[target_dim] = size_diff
      return orig_size

    for a in self.attrs:
      old_x = getattr(self, a)
      setattr(self, a, torch.cat((old_x, old_x.new_zeros(same_dim_except_target(old_x.size()))), target_dim))

  def expand_beam_dim(self, new_size):
    self.expand_at_dim(1, new_size)


class FixedStack(ExpandableStorage):
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
    self.stack_size = stack_size

    self.pointer = torch.zeros(batch_size, dtype=torch.long, device=device)  # word pointer
    self.top_position = torch.zeros(batch_size, dtype=torch.long, device=device)  # stack top position
    self.hiddens = initial_hidden[0].new_zeros(batch_size + (stack_size+1, hidden_size, num_layers), device=device)
    self.cells = initial_hidden[0].new_zeros(batch_size + (stack_size+1, hidden_size, num_layers), device=device)
    self.trees = initial_hidden[0].new_zeros(batch_size + (stack_size, input_size), device=device)

    if beam_size == 1:
      self.hiddens[:, 0] = initial_hidden[0]
      self.cells[:, 0] = initial_hidden[1]
    else:
      # Only fill zero-th beam position because we do not have other beam elems at beginning of search.
      self.hiddens[:, 0, 0] = initial_hidden[0]
      self.cells[:, 0, 0] = initial_hidden[1]

    self.nt_index = torch.zeros(batch_size + (stack_size,), dtype=torch.long, device=device)
    self.nt_ids = torch.zeros(batch_size + (stack_size,), dtype=torch.long, device=device)
    self.nt_index_pos = torch.tensor([-1], dtype=torch.long, device=device).expand(batch_size).clone() # default is -1 (0 means zero-dim exists)

    self.attrs = ['pointer', 'top_position', 'hiddens', 'cells', 'trees',
                  'nt_index', 'nt_ids', 'nt_index_pos']

  @property
  def beam_size(self):
    if self.hiddens.dim() == 5:
      return self.hiddens.size(1)
    else:
      return 1

  def reset_batch_index(self):
    # a necessary operation when expanding beam dimension.
    assert self.beam_size > 1
    device = self.batch_index[0].device
    self.batch_index = ((torch.arange(0, self.batch_size, dtype=torch.long, device=device)
                         .unsqueeze(1).expand(-1, self.beam_size).reshape(-1)),
                        torch.cat([torch.arange(0, self.beam_size, dtype=torch.long, device=device)
                                   for _ in range(self.batch_size)]))

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

  def do_shift(self, shift_batches, shifted_embs, subword_end_mask = None):
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
    self.trees[reduce_batches + (prev_nt_position-1,)] = new_child
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
    self.hiddens[self.batch_index + (pos,)] = new_hidden
    self.cells[self.batch_index + (pos,)] = new_cell

  def sort_by(self, sort_idx):
    """

    :param sort_idx: (batch_size, beam_size) or (batch_size)
    """

    def sort_tensor(tensor):
      _idx = sort_idx
      for i in range(sort_idx.dim(), tensor.dim()):
        _idx = _idx.unsqueeze(-1)
      return torch.gather(tensor, sort_idx.dim()-1, _idx.expand(tensor.size()))

    for a in self.attrs:
      old_x = getattr(self, a)
      setattr(self, a, sort_tensor(old_x))

  def move_beams(self, self_move_idxs, other, move_idxs):
    for a in self.attrs:
      getattr(self, a)[self_move_idxs] = getattr(other, a)[move_idxs]


class StackState(ExpandableStorage):
  def __init__(self, batch_size, beam_size, device):
    """Keep track of information about states that is strategy-dependent, including
    ncons_nts, for which how to update it will depend on the strategy.

    Structures other than FixedStack and StackState preserved in BeamItems would be
    strategy-invariant.

    """
    super(StackState, self).__init__()

    self.ncons_nts = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
    self.nopen_parens = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)

    self.attrs = ['ncons_nts', 'nopen_parens']

  def move_beams(self, self_idxs, source, source_idxs):
    self.ncons_nts[self_idxs] = source.ncons_nts[source_idxs]
    self.nopen_parens[self_idxs] = source.nopen_parens[source_idxs]

  def sort_by(self, sort_idx):
    self.ncons_nts = torch.gather(self.ncons_nts, 1, sort_idx)
    self.nopen_parens = torch.gather(self.nopen_parens, 1, sort_idx)

  def update_nt_counts(self, actions, action_dict, action_path = None):
    shift_idxs = (actions == action_dict.a2i['SHIFT']).nonzero(as_tuple=True)
    nt_idxs = (actions >= action_dict.nt_begin_id()).nonzero(as_tuple=True)
    reduce_idxs = (actions == action_dict.a2i['REDUCE']).nonzero(as_tuple=True)

    self.ncons_nts[shift_idxs] = 0
    self.nopen_parens[nt_idxs] += 1
    self.ncons_nts[nt_idxs] += 1
    self.nopen_parens[reduce_idxs] -= 1
    self.ncons_nts[reduce_idxs] = 0


class ActionPath(ExpandableStorage):
  def __init__(self, batch_size, beam_size, max_actions, device):
    super(ActionPath, self).__init__()
    self.actions = torch.full((batch_size, beam_size, max_actions), -1, dtype=torch.long, device=device)
    self.actions_pos = self.actions.new_zeros(batch_size, beam_size)
    self.attrs = ['actions', 'actions_pos']

  def prev_actions(self):
    return self.actions.gather(2, self.actions_pos.unsqueeze(-1)).squeeze(-1)  # (batch_size, beam_size)

  def nbest_parses(self, beam_widths, gen_ll, tgt_batch = None):
    widths = beam_widths.cpu().numpy()
    actions = self.actions.cpu().numpy()
    actions_pos = self.actions_pos.cpu().numpy()
    def tree_actions(batch, beam):
      return (actions[batch, beam, 1:actions_pos[batch, beam]+1].tolist(), gen_ll[batch, beam].item())

    def batch_actions(batch):
      return [tree_actions(batch, i) for i in range(widths[batch])]

    if tgt_batch is not None:
      return batch_actions(tgt_batch)
    else:
      return [batch_actions(b) for b in range(len(widths))]

  def move_beams(self, self_idxs, source, source_idxs):
    self.actions[self_idxs] = source.actions[source_idxs]
    self.actions_pos[self_idxs] = source.actions_pos[source_idxs]

  def sort_by(self, sort_idx):
    self.actions = torch.gather(self.actions, 1,
                                sort_idx.unsqueeze(-1).expand(self.actions.size()))
    self.actions_pos = torch.gather(self.actions_pos, 1, sort_idx)

  def add(self, actions, active_idxs):
    self.actions_pos[active_idxs] += 1
    self.actions[active_idxs + (self.actions_pos[active_idxs],)] = actions[active_idxs]


class BeamAdditionalScores(ExpandableStorage):
  """There is no additional scores for beam search."""
  def __init__(self):
    super(BeamAdditionalScores, self).__init__()

  def move_beams(self, self_idxs, source, source_idxs):
    pass

  def sort_by(self, sort_idx):
    pass

  def move_beams(self, self_idxs, source, source_idxs, new_additional):
    pass


class ParticleAdditionalScores(ExpandableStorage):
  def __init__(self, batch_size, beam_size, initial_K, device):
    super(ParticleAdditionalScores, self).__init__()
    self.disc_ll = torch.tensor([-float('inf')], device=device).expand(
      batch_size, beam_size).clone()
    self.K = torch.tensor([initial_K], dtype=torch.float, device=device).expand(
      batch_size, beam_size).clone()
    self.disc_ll[:, 0] = 0

    self.attrs = ['disc_ll', 'K']

  def move_beams(self, self_idxs, source, source_idxs):
    self.disc_ll[self_idxs] = source.disc_ll[source_idxs]
    self.K[self_idxs] = source.K[source_idxs]

  def sort_by(self, sort_idx):
    self.disc_ll = torch.gather(self.disc_ll, 1, sort_idx)
    self.K = torch.gather(self.K, 1, sort_idx)

  def move_beams(self, self_idxs, source, source_idxs, new_additional = ()):
    if len(new_additional) > 0:
      assert len(new_additional) == 2
      self.disc_ll[self_idxs] = new_additional[0]
      self.K[self_idxs] = new_additional[1]
    else:
      self.disc_ll[self_idxs] = source.disc_ll[source_idxs]
      self.K[self_idxs] = source.K[source_idxs]


class BeamItems(ExpandableStorage):
  def __init__(self, stack, stack_state, max_actions = 500, beam_is_empty = False,
               particle_filter = False, initial_K = 0):
    super(BeamItems, self).__init__()
    self.batch_size = stack.batch_size
    self.beam_size = stack.beam_size
    self.stack = stack
    self.stack_state = stack_state

    if particle_filter:
      assert initial_K > 0
      self.additional = ParticleAdditionalScores(
        self.batch_size, self.beam_size, initial_K, stack.hiddens.device)
    else:
      self.additional = BeamAdditionalScores()

    self.gen_ll = torch.tensor([-float('inf')], device=stack.hiddens.device).expand(
      self.batch_size, self.beam_size).clone()
    self.gen_ll[:, 0] = 0

    if beam_is_empty:
        # how many beam elements are active for each batch?
      self.beam_widths = self.gen_ll.new_zeros(self.batch_size, dtype=torch.long)
    else:
      self.beam_widths = self.gen_ll.new_ones(self.batch_size, dtype=torch.long)

    self.action_path = ActionPath(self.batch_size, self.beam_size, max_actions,
                                  self.beam_widths.device)

    self.finished = self.beam_widths.new_zeros((self.batch_size, self.beam_size), )

    self.attrs = ['gen_ll', 'finished']

  @property
  def ncons_nts(self):
    return self.stack_state.ncons_nts

  @property
  def nopen_parens(self):
    return self.stack_state.nopen_parens

  @property
  def actions(self):
    return self.action_path.actions

  @property
  def actions_pos(self):
    return self.action_path.actions_pos

  def prev_actions(self):
    return self.action_path.prev_actions()

  def nbest_parses(self, batch = None):
    return self.action_path.nbest_parses(self.beam_widths, self.gen_ll, batch)

  def shrink(self, size = -1):
    size = size if size > 0 else self.beam_size
    outside_beam_idx = (torch.arange(self.beam_size, device=self.gen_ll.device).unsqueeze(0) >=
                        self.beam_widths.unsqueeze(1)).nonzero(as_tuple=True)
    self.gen_ll[outside_beam_idx] = -float('inf')
    self.gen_ll, sort_idx = torch.sort(self.gen_ll, descending=True)
    self.additional.sort_by(sort_idx)
    self.stack.sort_by(sort_idx)
    self.stack_state.sort_by(sort_idx)
    self.action_path.sort_by(sort_idx)
    self.beam_widths = torch.min(self.beam_widths, self.beam_widths.new_tensor([size]))

  def active_idxs(self):
    """
    :return (batch_idxs, beam_idxs): All active idxs according to active beam sizes for each batch
                                     defined by self.beam_widths.
    """
    return self.active_idx_mask().nonzero(as_tuple=True)

  def active_idx_mask(self):
    order = torch.arange(self.beam_size, device=self.beam_widths.device)
    return order < self.beam_widths.unsqueeze(1)

  def clear(self):
    self.beam_widths[:] = 0

  def move_items_from(self, other, move_idxs, new_gen_ll = None,
                      additional = (), allow_expand=False):
    """

    :param other: BeamItems
    :param move_idxs: A pair of index tensors (for batch_index and beam_index)
    :param new_gen_ll: If not None, replace gen_ll of the target positions with this vector.
    :param additional: Tuple of vectors. If not empty, used to update AdditionalScores.
    """
    assert len(move_idxs) == 2  # hard-coded for beam search case.
    # This method internally presupposes that batch_index is sorted.
    assert torch.equal(move_idxs[0].sort()[0], move_idxs[0])
    move_batch_idxs, move_beam_idxs = move_idxs

    batch_numbers = bincount_and_supply(move_batch_idxs, self.batch_size)
    max_moved_beam_size = batch_numbers.max()
    new_beam_widths = self.beam_widths + batch_numbers  # (batch_size)
    if new_beam_widths.max() >= self.beam_size and allow_expand:
      # Try to expand when exceeding max beam size if allow_expand is set to True.
      self.resize(new_beam_widths.max() + 100)
    if new_beam_widths.max() >= self.beam_size:
      # The default case of handling beam widths exceeding max beam size, discarding
      # elements not fitted in self.beam_size.
      # This may be called even after the resize operation above because there is
      # an upperbound on the beam size.
      beam_idx_order = torch.arange(max_moved_beam_size, device=batch_numbers.device)
      sum_beam_idx_order = self.beam_widths.unsqueeze(1) + beam_idx_order
      move_idx_mask = sum_beam_idx_order < self.beam_size
      move_idx_mask = move_idx_mask.view(-1)[
        (beam_idx_order < batch_numbers.unsqueeze(1)).view(-1)]
      move_idxs = (move_idxs[0][move_idx_mask], move_idxs[1][move_idx_mask])
      move_batch_idxs, move_beam_idxs = move_idxs
      if new_gen_ll is not None:
        new_gen_ll = new_gen_ll[move_idx_mask]
      additional = [score[move_idx_mask] for score in additional]
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
    self._do_move_elements(other, self_move_idxs, move_idxs, new_gen_ll, additional)

    return self_move_idxs

  def resize(self, new_size = -1, max_size = 3000):
    if new_size <= 0:
      new_size = self.beam_size + 100
    new_size = min(new_size, max_size)
    if self.beam_size >= new_size:
      # Not expanding if current size already reaches the limit.
      return
    self.stack.expand_beam_dim(new_size)
    self.stack.reset_batch_index()
    self.stack_state.expand_beam_dim(new_size)
    self.additional.expand_beam_dim(new_size)
    self.action_path.expand_beam_dim(new_size)
    self.expand_beam_dim(new_size)
    self.beam_size = new_size

  def reconstruct(self, target_idxs, allow_expand=False):
    """

    Intuitively perform beam[:] = beam[target_idxs]. target_idxs contains duplicates so this would
    copy some elements across different idxs. A core function in beam search.
    """
    assert self.beam_widths.sum() > 0

    assert len(target_idxs) == 2  # hard-coded for beam search case.
    move_batch_idxs, move_beam_idxs = target_idxs
    self.beam_widths = bincount_and_supply(move_batch_idxs, self.batch_size)
    max_beam_widths = self.beam_widths.max()
    target_mask = None
    if max_beam_widths > self.beam_size and allow_expand:
      self.resize(max_beam_widths + 100)
      max_beam_widths = self.beam_widths.max()
    if max_beam_widths > self.beam_size:
      # need to shrink (may occur in particle filtering)
      beam_idx_order = torch.arange(max_beam_widths, device=target_idxs[0].device)
      target_mask = beam_idx_order.unsqueeze(0).expand(self.batch_size, -1) < self.beam_size
      target_mask = target_mask.view(-1)[(beam_idx_order < self.beam_widths.unsqueeze(1)).view(-1)]
      target_idxs = (target_idxs[0][target_mask], target_idxs[1][target_mask])
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

    return self_move_idxs, target_mask

  def marginal_probs(self):
    active_idx_mask = self.active_idx_mask()
    self.gen_ll[active_idx_mask != 1] = -float('inf')
    return torch.logsumexp(self.gen_ll, 1)

  def _do_move_elements(self, source, self_idxs, source_idxs, new_gen_ll = None,
                        new_additional = ()):
    self.gen_ll[self_idxs] = new_gen_ll if new_gen_ll is not None else source.gen_ll[source_idxs]
    self.additional.move_beams(self_idxs, source.additional, source_idxs, new_additional)
    self.stack_state.move_beams(self_idxs, source.stack_state, source_idxs)
    self.stack.move_beams(self_idxs, source.stack, source_idxs)
    self.action_path.move_beams(self_idxs, source.action_path, source_idxs)

  def do_action(self, actions, action_dict):
    # We need to use "unupdated" action_path for updating stack_state.
    self.stack_state.update_nt_counts(actions, action_dict, self.action_path)
    self.action_path.add(actions, self.active_idxs())


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
    self.dropout = nn.Dropout(dropout, inplace=True)

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

  def forward(self, word_vecs, actions, stack, subword_end_mask = None):
    """
    Similar to update_stack_rnn.

    :param word_vecs: (batch_size, sent_len, input_size)
    :param actions: (batch_size, 1)
    """

    reduce_batches = (actions == self.action_dict.a2i['REDUCE']).nonzero(as_tuple=True)
    nt_batches = (actions >= self.action_dict.nt_begin_id()).nonzero(as_tuple=True)
    shift_batches = (actions == self.action_dict.a2i['SHIFT']).nonzero(as_tuple=True)

    new_input = stack.trees.new_zeros(stack.hiddens.size()[:-3] + (self.input_size,))

    # First fill in trees. Then, gather those added elements in a column, which become
    # the input to stack_rnn.
    if shift_batches[0].size(0) > 0:
      shift_idx = stack.pointer[shift_batches].view(-1, 1, 1).expand(-1, 1, word_vecs.size(-1))
      shifted_embs = torch.gather(word_vecs[shift_batches[0]], 1, shift_idx).squeeze(1).to(new_input.dtype)
      stack.do_shift(shift_batches, shifted_embs, subword_end_mask)
      new_input[shift_batches] = shifted_embs

    if nt_batches[0].size(0) > 0:
      nt_ids = (actions[nt_batches] - self.action_dict.nt_begin_id())
      nt_embs = self.nt_emb(nt_ids).to(new_input.dtype)
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
      new_input[reduce_batches] = new_child.to(new_input.dtype)

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

    self.dropout = nn.Dropout(dropout, inplace=True)
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

  def forward(self, x, actions, initial_stack = None, stack_size_bound = -1,
              subword_end_mask = None):
    assert isinstance(x, torch.Tensor)
    assert isinstance(actions, torch.Tensor)

    if stack_size_bound <= 0:
      stack_size = max(100, x.size(1) + 10)
    else:
      stack_size = stack_size_bound
    stack = self.build_stack(x, stack_size)
    word_vecs = self.emb(x)
    action_contexts = self.unroll_states(stack, word_vecs, actions, subword_end_mask)

    a_loss, _ = self.action_loss(actions, self.action_dict, action_contexts)
    w_loss, _ = self.word_loss(x, actions, self.action_dict, action_contexts)
    loss = (a_loss.sum() + w_loss.sum())
    return loss, a_loss, w_loss, stack

  def unroll_states(self, stack, word_vecs, actions, subword_end_mask = None):
    hs = word_vecs.new_zeros(actions.size(1), word_vecs.size(0), self.hidden_size)
    hs[0] = stack.hidden_head()[:, :, -1]
    for step in range(actions.size(1)-1):
      h = self.rnng(word_vecs, actions[:, step], stack, subword_end_mask)  # (batch_size, input_size)
      hs[step+1] = h
    hs = self.rnng.output(hs.transpose(1, 0).contiguous())  # (batch_size, action_len, input_size)
    return hs

  def build_stack(self, x, stack_size = 80):
    initial_hidden = self.rnng.get_initial_hidden(x)
    return FixedStack(initial_hidden, stack_size, self.input_size)

  def action_loss(self, actions, action_dict, hiddens):
    assert hiddens.size()[:2] == actions.size()
    actions = actions.view(-1)
    hiddens = hiddens.view(actions.size(0), -1)

    action_mask = actions != action_dict.padding_idx
    idx = action_mask.nonzero(as_tuple=False).squeeze(1)
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
    idx = action_mask.nonzero(as_tuple=False).squeeze(1)
    hiddens = hiddens[idx]

    x = x.view(-1)
    x = x[x != self.padding_idx]
    assert x.size(0) == hiddens.size(0)
    logit = self.vocab_mlp(hiddens)
    loss = self.word_criterion(logit, x)
    return loss, logit

  def word_sync_beam_search(self, x, subword_end_mask, beam_size, word_beam_size = 0,
                            shift_size = 0, stack_size_bound = 100,
                            return_beam_history = False):
    self.eval()
    sent_lengths = (x != self.padding_idx).sum(dim=1)
    if (hasattr(self.rnng.composition, 'batch_index') and
        self.rnng.composition.batch_index.size(0) < x.size(0)*beam_size):
      # The maximum number may be set by assuming training setting only.
      # Here we reset to the maximum number by beam search.
      self.rnng.composition.batch_index = torch.arange(0, x.size(0)*beam_size, device=x.device)

    if word_beam_size <= 0:
      word_beam_size = beam_size

    beam, word_completed_beam = self.build_beam_items(x, beam_size, shift_size,
                                                      stack_size_bound=stack_size_bound)
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    parses = [None] * x.size(0)
    surprisals = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      forced_completions = x.new_zeros(x.size(0), dtype=torch.long)
      bucket_i = 0

      def word_finished_batches():
        return ((beam.beam_widths == 0) +  # Empty beam means no action remains (finished).
                (word_completed_beam.beam_widths >= word_completed_beam.beam_size) +
                ((word_completed_beam.beam_widths - forced_completions) >= beam_size))

      finished_batches = word_finished_batches()

      while not finished_batches.all():
        added_forced_completions = self.beam_step(
          x, subword_end_mask, sent_lengths, word_vecs, pointer, beam,
          word_completed_beam, shift_size)
        forced_completions += added_forced_completions
        finished_batches = word_finished_batches()
        beam.beam_widths[finished_batches.nonzero(as_tuple=True)] = 0  # inactive word-finished batches.
        bucket_i += 1

      self.finalize_word_completed_beam(
        x, subword_end_mask, sent_lengths, word_vecs, pointer, beam,
        word_completed_beam, word_beam_size)

      marginal = beam.marginal_probs()
      for b, s in enumerate(marginal.cpu().detach().numpy()):
        word_marginal_ll[b].append(s)

      for b, length in enumerate(sent_lengths.cpu().detach()):
        if length == pointer:  # finished
          for i in range(0, length):
            surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))
          parses[b] = beam.nbest_parses(b)
          beam.beam_widths[b] = 0  # inactivate finished batch

    ret = (parses, surprisals)
    return ret

  def beam_step(self, x, subword_end_mask, sent_lengths, word_vecs, pointer, beam,
                word_completed_beam, shift_size):
    beam_size = beam.beam_size
    successors, word_completed_successors, added_forced_completions \
      = self.get_successors(x, subword_end_mask, sent_lengths, pointer, beam, beam_size, shift_size)

    # tuple of ((batch_idxs, beam_idxs), next_actions, total_scores)
    assert len(successors) == len(word_completed_successors) == 3

    if word_completed_successors[0][0].size(0) > 0:
      comp_idxs = tuple(word_completed_successors[0][:2])
      # Add elements to word_completed_beam
      # This assumes that returned scores are total scores rather than the current action scores.
      moved_idxs = word_completed_beam.move_items_from(
        beam, comp_idxs, new_gen_ll=word_completed_successors[2])

    new_beam_idxs, _ = beam.reconstruct(successors[0][:2])
    beam.gen_ll[new_beam_idxs] = successors[2]
    actions = successors[1].new_full((x.size(0), beam_size), self.action_dict.padding_idx)
    actions[new_beam_idxs] = successors[1]
    self.rnng(word_vecs, actions, beam.stack, subword_end_mask)
    beam.do_action(actions, self.action_dict)

    return added_forced_completions

  def variable_beam_step(self, x, subword_end_mask, sent_lengths, word_vecs, pointer,
                         beam, word_completed_beam, K):
    successors, word_completed_successors \
      = self.get_successors_by_particle_filter(x, subword_end_mask, sent_lengths, pointer, beam)
    # successors: ((batch_idx, beam_idx), action, gen_ll, disc_ll, K)
    if word_completed_successors[0][0].size(0) > 0:
      comp_idxs = tuple(word_completed_successors[0])
      word_completed_beam.move_items_from(
        beam,
        comp_idxs,
        new_gen_ll=word_completed_successors[2],
        additional=word_completed_successors[3:5],
        allow_expand=True)

    new_beam_idxs, target_mask = beam.reconstruct(successors[0], allow_expand=True)
    if target_mask is not None:
      successors = list(successors)
      for i in range(1, 5):
        successors[i] = successors[i][target_mask]
    beam.gen_ll[new_beam_idxs] = successors[2]
    beam.additional.move_beams(new_beam_idxs, None, None, successors[3:5])
    actions = successors[1].new_full((x.size(0), beam.beam_size),
                                     self.action_dict.padding_idx)
    actions[new_beam_idxs] = successors[1]

    self.rnng(word_vecs, actions, beam.stack, subword_end_mask)
    beam.do_action(actions, self.action_dict)

  def finalize_word_completed_beam(
      self, x, subword_end_mask, sent_lengths, word_vecs, pointer, beam,
      word_completed_beam, word_beam_size):
    beam_size = word_completed_beam.beam_size
    word_completed_beam.shrink(word_beam_size)
    word_end_actions = x.new_full((x.size(0), beam_size), self.action_dict.padding_idx)
    # active_idx = word_completed_beam.active_idxs()
    # if pointer < x.size(1):  # do shift
    #   word_end_actions[active_idx] = self.action_dict.a2i['SHIFT']
    # else:
    #   word_end_actions[active_idx] = self.action_dict.finish_action()
    active_idx_mask = word_completed_beam.active_idx_mask()
    shift_beam_idx_mask = (pointer < sent_lengths).unsqueeze(1) * active_idx_mask
    finish_beam_idx_mask = (pointer == sent_lengths).unsqueeze(1) * active_idx_mask
    word_end_actions[shift_beam_idx_mask] = self.action_dict.a2i['SHIFT']
    word_end_actions[finish_beam_idx_mask] = self.action_dict.finish_action()

    self.rnng(word_vecs, word_end_actions, word_completed_beam.stack, subword_end_mask)
    word_completed_beam.do_action(word_end_actions, self.action_dict)

    beam.clear()
    active_idx = active_idx_mask.nonzero(as_tuple=True)
    beam.move_items_from(word_completed_beam, active_idx)
    word_completed_beam.clear()

  def build_beam_items(self, x, beam_size, shift_size, particle=False, K=0,
                       stack_size_bound=100):
    #stack_size = max(100, x.size(1) + 10)
    if stack_size_bound <= 0:
      stack_size = max(100, x.size(1) + 10)
    else:
      #stack_size = min(int(x.size(1)*2.5), stack_size_bound)
      stack_size = min(x.size(1) + 20, stack_size_bound)
    stack_size = math.ceil(stack_size / 8) * 8  # force to be multiple of 8.
    initial_hidden = self.rnng.get_initial_hidden(x)
    stack_unfinished, state_unfinished = self.new_beam_stack_with_state(
      initial_hidden, stack_size, beam_size)
    # The rationale behind (+shift_size*5) for beam size for finished BeamItems is
    # that # steps between words would probably be ~5 in most cases. Forcing to save shifts
    # after this many steps seems to be unnecessary.
    stack_word_finished, state_word_finished = self.new_beam_stack_with_state(
      initial_hidden, stack_size, min(beam_size * 2, beam_size + shift_size*5))
    max_actions = max(100, x.size(1) * 5)
    return (BeamItems(stack_unfinished, state_unfinished, max_actions,
                      False, particle, K),
            BeamItems(stack_word_finished, state_word_finished, max_actions,
                      True, particle, K))

  def new_beam_stack_with_state(self, initial_hidden, stack_size, beam_size):
    stack = FixedStack(initial_hidden, stack_size, self.input_size, beam_size)
    stack_state = StackState(initial_hidden[0].size(0), beam_size, initial_hidden[0].device)
    return stack, stack_state

  def variable_beam_search(self, x, subword_end_mask, K, original_reweight=False, stack_size_bound=100):
    self.eval()
    sent_lengths = (x != self.padding_idx).sum(dim=1)
    #max_beam_size = min(K//2, 1000)
    max_beam_size = 100
    if (hasattr(self.rnng.composition, 'batch_index') and
        self.rnng.composition.batch_index.size(0) < x.size(0)*max_beam_size):
      # The maximum number may be set by assuming training setting only.
      # Here we reset to the maximum number by beam search.
      self.rnng.composition.batch_index = torch.arange(
        0, x.size(0)*max_beam_size, device=x.device)

    beam, word_completed_beam = self.build_beam_items(x, max_beam_size, 0, True, K,
                                                      stack_size_bound=stack_size_bound)
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    parses = [None] * x.size(0)
    surprisals = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      bucket_i = 0
      while not ((beam.beam_widths == 0) +
                 (word_completed_beam.beam_widths == word_completed_beam.beam_size-1)).all():
        self.variable_beam_step(
          x, subword_end_mask, sent_lengths, word_vecs, pointer, beam, word_completed_beam, K)
        bucket_i += 1

      self.finalize_particle_beam(
        x, subword_end_mask, sent_lengths, word_vecs, pointer, beam, word_completed_beam, K,
        original_reweight=original_reweight)

      marginal = beam.marginal_probs()
      for b, s in enumerate(marginal.cpu().detach().numpy()):
        word_marginal_ll[b].append(s)

      # This shrink is for sorting by gen_ll as a preparation before choosing nbest_parses.
      beam.shrink()
      for b, length in enumerate(sent_lengths.cpu().detach()):
        if length == pointer:  # finished
          for i in range(0, length):
            surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))
          parses[b] = beam.nbest_parses(b)
          beam.beam_widths[b] = 0  # inactivate finished batch

    ret = (parses, surprisals)
    return ret

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

  def get_successors(self, x, subword_end_mask, sent_lengths, pointer, beam, beam_size, shift_size):
    if pointer < x.size(1):
      next_x = x[:, pointer]
    else:
      next_x = None

    invalid_action_mask = self.invalid_action_mask(beam, sent_lengths, subword_end_mask)  # (total beam size, n_actions)

    log_probs = self.action_log_probs(beam.stack, invalid_action_mask, next_x)  # (batch, beam, n_actions)
    # scores for inactive beam items (outside active_idx) are -inf on log_probs so we need
    # not worry about values in gen_ll outside active_idx.
    log_probs += beam.gen_ll.unsqueeze(-1)

    return self.scores_to_successors(x, sent_lengths, pointer, beam, log_probs, beam_size, shift_size)

  def get_successors_by_particle_filter(self, x, subword_end_mask, sent_lengths, pointer, beam):
    if pointer < x.size(1):
      next_x = x[:, pointer]
    else:
      next_x = None

    invalid_action_mask = self.invalid_action_mask(beam, sent_lengths, subword_end_mask)  # (total beam size, n_actions)

    # nan is already transformed to -inf.
    log_probs, disc_log_probs = self.action_log_probs(
      beam.stack, invalid_action_mask, next_x, return_disc_probs=True)
    new_K = (disc_log_probs.exp() * beam.additional.K.unsqueeze(-1)).round_()
    new_K = new_K.view(new_K.size(0), -1)  # (batch, beam*n_actions)

    log_probs += beam.gen_ll.unsqueeze(-1)
    disc_log_probs += beam.additional.disc_ll.unsqueeze(-1)

    num_actions = log_probs.size(2)
    idx = torch.arange(0, new_K.size(1), device=new_K.device).unsqueeze(0).expand(new_K.size())

    log_probs = log_probs.view(log_probs.size(0), -1)
    disc_log_probs = disc_log_probs.view(disc_log_probs.size(0), -1)
    valid_action_mask = new_K > 0.0
    beam_id = idx // num_actions
    action_id = idx % num_actions

    end_action_mask = (
      ((pointer < sent_lengths).unsqueeze(1) * action_id == self.action_dict.a2i['SHIFT']) +
      ((pointer == sent_lengths).unsqueeze(1) * self._parse_finish_mask(beam, action_id, beam_id))
    )

    no_end_action_mask = (end_action_mask != 1) * valid_action_mask
    end_action_mask = end_action_mask * valid_action_mask

    def mask_to_successors(action_mask):
      masked_idx = action_mask.nonzero(as_tuple=True)
      return ((masked_idx[0], beam_id[masked_idx]),
              action_id[masked_idx],
              log_probs[masked_idx],
              disc_log_probs[masked_idx],
              new_K[masked_idx])

    return (mask_to_successors(no_end_action_mask),
            mask_to_successors(end_action_mask))

  def finalize_particle_beam(self, x, subword_end_mask, sent_lengths, word_vecs, pointer,
                             beam, word_completed_beam, K, original_reweight=False):
    # Filtered elements are moved to beam from word_completed_beam
    # (but no action performed yet)
    self.reweight_and_filter_particles(beam, word_completed_beam, K, original_reweight)

    word_end_actions = x.new_full((x.size(0), beam.beam_size),
                                  self.action_dict.padding_idx)
    active_idx_mask = beam.active_idx_mask()
    shift_beam_idx_mask = (pointer < sent_lengths).unsqueeze(1) * active_idx_mask
    finish_beam_idx_mask = (pointer == sent_lengths).unsqueeze(1) * active_idx_mask
    word_end_actions[shift_beam_idx_mask] = self.action_dict.a2i['SHIFT']
    word_end_actions[finish_beam_idx_mask] = self.action_dict.finish_action()
    self.rnng(word_vecs, word_end_actions, beam.stack, subword_end_mask)
    beam.do_action(word_end_actions, self.action_dict)

  def reweight_and_filter_particles(
      self, beam, word_completed_beam, K, original_reweight=False):
    if original_reweight:
      # Try to avoid fp16 before logsumexp and exp, which may be sensitive to precision.
      log_weights = (word_completed_beam.additional.K.log().float() +
                     word_completed_beam.gen_ll.float() -
                     word_completed_beam.disc_ll.float())
    else:
      log_weights = word_completed_beam.gen_ll.float()
    log_weights[word_completed_beam.active_idx_mask() != 1] = -float('inf')

    denom = torch.logsumexp(log_weights, 1)  # (batch_size,)
    new_Ks = ((log_weights - denom.unsqueeze(-1)).exp_() * K).round_()  # (batch, beam)

    new_active_idx = new_Ks.nonzero(as_tuple=True)
    word_completed_beam.additional.K = new_Ks  # reweight
    beam.clear()
    beam.move_items_from(word_completed_beam, new_active_idx, allow_expand=True)
    word_completed_beam.clear()

  def action_log_probs(self, stack, invalid_action_mask, next_x = None, return_disc_probs = False):
    """

    :param stack: FixedStack
    :param invalid_action_mask: (batch_size, beam_size, num_actions)  (inactive beams are entirely masked.)
    :param next_x: (batch_size) to be shifted token ids
    """
    hiddens = self.rnng.output(stack.hidden_head()[:, :, -1])  # (beam*batch, hidden_size)
    # fp16 is cancelled here before softmax (I want to keep precision in final probablities).
    action_logit = self.action_mlp(hiddens).view(invalid_action_mask.size()).float()  # (beam, batch, num_actions)
    action_logit[invalid_action_mask] = -float('inf')

    log_probs = F.log_softmax(action_logit, -1)  # (batch_size, beam_size, num_actions)
    log_probs[torch.isnan(log_probs)] = -float('inf')
    if return_disc_probs:
      disc_log_probs = log_probs.clone()

    if next_x is not None:  # shift is valid for next action
      word_logit = self.vocab_mlp(hiddens).float()  # (batch*beam, vocab_size)
      word_logit[:, self.padding_idx] = -float('inf')
      shift_idx = self.action_dict.a2i['SHIFT']
      next_x = next_x.unsqueeze(1).expand(-1, log_probs.size(1)).clone().view(-1)  # (batch*beam)
      word_log_probs = self.word_criterion(word_logit, next_x) * -1.0  # (batch_size*beam_size)
      word_log_probs = word_log_probs.view(log_probs.size(0), log_probs.size(1))
      log_probs[:, :, shift_idx] += word_log_probs

    if return_disc_probs:
      return (log_probs, disc_log_probs)
    else:
      return log_probs

  def invalid_action_mask(self, beam, sent_lengths, subword_end_mask):
    """Return a tensor where mask[i,j,k]=True means action k is not allowed for beam (i,j).
    """
    action_order = torch.arange(self.num_actions, device=beam.nopen_parens.device)

    sent_lengths = sent_lengths.unsqueeze(-1)  # add beam dimension

    prev_pointer = beam.stack.pointer - 1
    prev_pointer[prev_pointer == -1] = 0
    prev_is_subword_mask = ((beam.stack.pointer > 0) *
                            (subword_end_mask.gather(1, prev_pointer) == 0))
    # reduce_mask[i,j,k]=True means k is a not allowed reduce action for (i,j).
    reduce_mask = (action_order == self.action_dict.a2i['REDUCE']).view(1, 1, -1)
    reduce_mask = ((((beam.nopen_parens == 1) * (beam.stack.pointer < sent_lengths)) +
                    # prev is nt => cannot reduce immediately after nt
                    (beam.prev_actions() >= self.action_dict.nt_begin_id()) +
                    (beam.stack.top_position < 2) +
                    # only shift is allowed when prev is subword
                    prev_is_subword_mask).unsqueeze(-1) *
                   reduce_mask)

    # nt_mask[i,j,k]=True means k is a not allowed nt action for (i,j).
    nt_mask = (action_order >= self.action_dict.nt_begin_id()).view(1, 1, -1)
    nt_mask = (((beam.nopen_parens >= self.max_open_nts) +
                (beam.ncons_nts >= self.max_cons_nts) +
                # Check the storage of beam.actions, which is bounded beforehand.
                # Theoretically +1 seems sufficient (for rhs); extra +2 is for saving cases
                # where other actions (reduce/shift) are prohibited for some reasons.
                (beam.actions.size(2) - beam.actions_pos < (
                  sent_lengths - beam.stack.pointer + beam.nopen_parens + 3)) +
                # Check the storage of fixed stack size (we need minimally two additional
                # elements to process arbitrary future structure).
                (beam.stack.top_position >= beam.stack.stack_size-2) +
                # only shift is allowed when prev is subword
                prev_is_subword_mask).unsqueeze(-1) *
               nt_mask)

    shift_mask = (action_order == self.action_dict.a2i['SHIFT']).view(1, 1, -1)
    shift_mask = (beam.stack.top_position >= beam.stack.stack_size-1).unsqueeze(-1) * shift_mask

    # all actions other than nt are invalid;
    # except_nt_mask[i,j,k]=True means k (not nt) is not allowed for (i,j).
    except_nt_mask = (action_order < self.action_dict.nt_begin_id()).view(1, 1, -1)
    except_nt_mask = (beam.nopen_parens == 0).unsqueeze(-1) * except_nt_mask

    except_reduce_mask = (action_order != self.action_dict.a2i['REDUCE']).view(1, 1, -1)
    except_reduce_mask = (beam.stack.pointer == sent_lengths).unsqueeze(-1) * except_reduce_mask

    pad_mask = (action_order == self.action_dict.padding_idx).view(1, 1, -1)
    finished_mask = ((beam.stack.pointer == sent_lengths) * (beam.nopen_parens == 0)).unsqueeze(-1)
    beam_width_mask = (torch.arange(beam.beam_size, device=reduce_mask.device).unsqueeze(0) >=
                       beam.beam_widths.unsqueeze(1)).unsqueeze(-1)

    return (reduce_mask + nt_mask + shift_mask + except_nt_mask + except_reduce_mask +
            pad_mask + finished_mask + beam_width_mask)

  def stack_top_h(self, states):
    return torch.stack([state.hiddens[-1][:, -1] for state in states], dim=0)

  def scores_to_successors(self, x, sent_lengths, pointer, beam, total_scores, beam_size, shift_size):
    num_actions = total_scores.size(2)
    total_scores = total_scores.view(total_scores.size(0), -1)
    sorted_scores, sort_idx = torch.sort(total_scores, descending=True)

    beam_id = sort_idx // num_actions
    action_id = sort_idx % num_actions

    valid_action_mask = sorted_scores != -float('inf')

    end_action_mask = (
      ((pointer < sent_lengths).unsqueeze(1) * action_id == self.action_dict.a2i['SHIFT']) +
      ((pointer == sent_lengths).unsqueeze(1) * self._parse_finish_mask(beam, action_id, beam_id))
    )

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

  def _parse_finish_mask(self, beam, action_id, beam_id):
    pre_final_mask = beam.nopen_parens.gather(1, beam_id) == 1
    end_action_mask = action_id == self.action_dict.finish_action()
    end_action_mask = end_action_mask * pre_final_mask
    return end_action_mask

