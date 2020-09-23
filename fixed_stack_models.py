import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import *
from torch.distributions import Bernoulli
import itertools


# class MultiLayerLSTMCell(nn.Module):
#   def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0):
#     super(MultiLayerLSTMCell, self).__init__()
#     self.lstm = nn.ModuleList()
#     self.lstm.append(nn.LSTMCell(input_size, hidden_size))
#     for i in range(num_layers-1):
#       self.lstm.append(nn.LSTMCell(hidden_size, hidden_size))
#     self.num_layers = num_layers
#     self.dropout = dropout
#     self.dropout_layer = nn.Dropout(dropout)

#   def forward(self, input, prev):
#     """

#     :param input: (batch_size, input_size)
#     :param prev: tuple of (h0, c0), each has size (batch, hidden_size, num_layers)
#     """

#     next_hidden = []
#     next_cell = []

#     if prev is None:
#         prev = (
#             input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0),
#             input.new(input.size(0), self.lstm[0].hidden_size, self.num_layers).fill_(0))

#     for i in range(self.num_layers):
#       prev_hidden_i = prev[0][:, :, i]
#       prev_cell_i = prev[1][:, :, i]
#       if i == 0:
#         next_hidden_i, next_cell_i = self.lstm[i](input, (prev_hidden_i, prev_cell_i))
#       else:
#         input_im1 = self.dropout_layer(input_im1)
#         next_hidden_i, next_cell_i = self.lstm[i](input_im1, (prev_hidden_i, prev_cell_i))
#       next_hidden += [next_hidden_i]
#       next_cell += [next_cell_i]
#       input_im1 = next_hidden_i

#     next_hidden = torch.stack(next_hidden).permute(1, 2, 0)
#     next_cell = torch.stack(next_cell).permute(1, 2, 0)

#     return next_hidden, next_cell


class MultiLayerLSTMCell(nn.Module):
  def __init__(self, i_dim, h_dim, num_layers, bias=True, dropout=0):
    super(MultiLayerLSTMCell, self).__init__()
    self.i_dim = i_dim
    self.h_dim = h_dim
    self.num_layers = num_layers
    self.linears = nn.ModuleList([nn.Linear(h_dim + i_dim, h_dim*4) if l == 0 else
                                  nn.Linear(h_dim*2, h_dim*4) for l in range(num_layers)])
    self.dropout = dropout
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, x, prev_h = None):
    if prev_h is None:
      prev_h = (x.new(x.size(0), self.h_dim, self.num_layers).fill_(0),
                x.new(x.size(0), self.h_dim, self.num_layers).fill_(0))

    next_hidden = []
    next_cell = []
    for l in range(self.num_layers):
      input = x if l == 0 else next_hidden[l-1]
      if l > 0 and self.dropout > 0:
        input = self.dropout_layer(input)
      prev_hidden_i = prev_h[0][:, :, l]
      prev_cell_i = prev_h[1][:, :, l]
      concat = torch.cat([input, prev_hidden_i], 1)  # (batch_size, h_dim*2) or (batch_size, h_dim+i_dim)
      all_sum = self.linears[l](concat)  # (batch_size, h_dim*4)
      i, f, o, g = all_sum.split(self.h_dim, 1)  # (batch_size, h_dim) for each
      c = torch.sigmoid(f)*prev_cell_i + torch.sigmoid(i)*torch.tanh(g)
      h = torch.sigmoid(o)*torch.tanh(c)
      next_hidden.append(h)
      next_cell.append(c)

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
    len_mask = length_to_mask(ch_lengths)
    logit[len_mask != 1] = -float('inf')
    attn = F.softmax(logit)
    weighted_child = (h*attn.unsqueeze(-1)).sum(1)
    weighted_child = self.dropout(weighted_child)

    nt2 = self.nt_emb2(nt_id)  # (batch_size, w_dim)
    gate_input = torch.cat([nt2, weighted_child], dim=-1)
    g = self.gate(gate_input)  # (batch_size, w_dim)
    c = g*nt2 + (1-g)*weighted_child  # (batch_size, w_dim)

    return self.output(c), attn, g


class FixedStack:
  def __init__(self, initial_hidden, stack_size, input_size):
    batch_size = initial_hidden[0].size(0)
    device = initial_hidden[1].device
    hidden_size = initial_hidden[0].size(-2)
    num_layers = initial_hidden[0].size(-1)

    self.pointer = torch.zeros(batch_size, dtype=torch.long, device=device)  # word pointer
    self.top_position = torch.zeros(batch_size, dtype=torch.long, device=device)  # stack top position
    self.hiddens = torch.zeros(batch_size, stack_size+1, hidden_size, num_layers, device=device)
    self.cells = torch.zeros(batch_size, stack_size+1, hidden_size, num_layers, device=device)
    self.trees = torch.zeros(batch_size, stack_size, input_size, device=device)

    self.batch_index = torch.arange(0, batch_size, dtype=torch.long, device=device)

    self.hiddens[self.batch_index, 0, :, :] = initial_hidden[0].float()
    self.cells[self.batch_index, 0, :, :] = initial_hidden[1].float()

    self.nt_index = torch.zeros(batch_size, stack_size, dtype=torch.long, device=device)
    self.nt_ids = torch.zeros(batch_size, stack_size, dtype=torch.long, device=device)
    self.nt_index_pos = torch.tensor([-1]*batch_size, dtype=torch.long, device=device) # default is -1 (0 means zero-dim exists)

    self.nopen_parens = [0] * batch_size
    self.ncons_nts = [0] * batch_size

  def hidden_head(self, offset = 0, batches = None):
    assert offset >= 0
    if batches is None:
      return self.hiddens[self.batch_index, self.top_position-offset, :, :]  # (batch_size, hidden_size, num_layers)
    else:
      return self.hiddens[batches, self.top_position[batches]-offset, :, :]

  def cell_head(self, offset = 0, batches = None):
    assert offset >= 0
    if batches is None:
      return self.cells[self.batch_index, self.top_position-offset, :, :]  # (batch_size, hidden_size, num_layers)
    else:
      return self.cells[batches, self.top_position[batches]-offset, :, :]

  def do_shift(self, shift_batches, shifted_embs):
    self.trees[shift_batches, self.top_position[shift_batches], :] = shifted_embs
    self.pointer[shift_batches] = self.pointer[shift_batches] + 1  # part of do_action is done here
    self.top_position[shift_batches] = self.top_position[shift_batches] + 1

  def do_nt(self, nt_batches, nt_embs, nt_ids):
    self.trees[nt_batches, self.top_position[nt_batches], :] = nt_embs

    self.nt_index_pos[nt_batches] = self.nt_index_pos[nt_batches] + 1
    self.nt_ids[nt_batches, self.nt_index_pos[nt_batches]] = nt_ids
    self.top_position[nt_batches] = self.top_position[nt_batches] + 1
    self.nt_index[nt_batches, self.nt_index_pos[nt_batches]] = self.top_position[nt_batches]

  def do_reduce(self, reduce_batches, new_child):
    prev_nt_position = self.nt_index[reduce_batches, self.nt_index_pos[reduce_batches]]
    self.trees[reduce_batches, prev_nt_position-1] = new_child.float()
    self.nt_index_pos[reduce_batches] = self.nt_index_pos[reduce_batches] - 1
    self.top_position[reduce_batches] = prev_nt_position

  def collect_reduced_children(self, reduce_batches):
    # Let us collect:
    #   reduced_nt_ids: (reduce_batch_size, 1)
    #   reduced_nts: (reduce_batch_size, input_size)
    #   reduced_children: (reduce_batch_size, max_child_len (padded), input_size)
    #   child_length: (reduce_batch_size, 1)
    nt_index_pos = self.nt_index_pos[reduce_batches]
    prev_nt_position = self.nt_index[reduce_batches, nt_index_pos]
    reduced_nt_ids = self.nt_ids[reduce_batches, nt_index_pos]
    reduced_nts = self.trees[reduce_batches, prev_nt_position-1]
    child_length = self.top_position[reduce_batches] - prev_nt_position
    max_ch_length = child_length.max()

    child_idx = prev_nt_position.unsqueeze(1) + torch.arange(max_ch_length, device=prev_nt_position.device)
    child_idx[child_idx >= self.trees.size(1)] = self.trees.size(1) - 1  # ceiled at maximum stack size (exceeding this may occur for some batches, but those should be ignored safely.)
    child_idx = child_idx.unsqueeze(2).expand(-1, -1, self.trees.size(-1))
    reduced_children = torch.gather(self.trees[reduce_batches], 1, child_idx)
    return reduced_children, child_length, reduced_nts, reduced_nt_ids

  def update_hidden(self, new_hidden, new_cell):
    pos = self.top_position.clone()
    self.hiddens[self.batch_index, pos, :, :] = new_hidden.float()
    self.cells[self.batch_index, pos, :, :] = new_cell.float()

  def reset_stack(self):
    # may be useful for reusing stack.
    self.pointer[:] = 0
    self.nt_index_pos[:] = -1

    self.nopen_parens = [0] * batch_size
    self.ncons_nts = [0] * batch_size

  def update_nt_counts(self, actions, action_dict):
    for i, a in enumerate(actions.cpu()):
      a = a.item()
      if action_dict.is_shift(a):
        self.ncons_nts[i] = 0
      elif action_dict.is_nt(a):
        self.nopen_parens[i] += 1
        self.ncons_nts[i] += 1
      elif action_dict.is_reduce(a):
        self.nopen_parens[i] -= 1
        self.ncons_nts[i] = 0


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
               attention_composition,
               nt_emb,
               stack_rnn,
               stack_to_hidden,
               composition,
               initial_emb
  ):
    super(RNNGCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # self.dropout = nn.Dropout(dropout)

    ####
    # self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts(), input_size), self.dropout)
    # self.stack_rnn = MultiLayerLSTMCell(input_size, hidden_size, num_layers, dropout=dropout)
    # self.output = nn.Sequential(self.dropout, nn.Linear(hidden_size, input_size), nn.ReLU())
    # self.composition = (AttentionComposition(input_size, dropout, self.action_dict.num_nts())
    #                     if attention_composition else
    #                     LSTMComposition(input_size, dropout))

    # self.initial_emb = nn.Sequential(nn.Embedding(1, input_size), self.dropout)
    self.nt_emb = nt_emb
    self.stack_rnn = stack_rnn
    self.output = stack_to_hidden
    self.composition = composition
    self.initial_emb = initial_emb
    ####

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

    reduce_batches = (actions == self.action_dict.a2i['REDUCE']).nonzero().squeeze(1)
    nt_batches = (actions >= self.action_dict.nt_begin_id()).nonzero().squeeze(1)
    shift_batches = (actions == self.action_dict.a2i['SHIFT']).nonzero().squeeze(1)

    new_input = word_vecs.new_zeros(word_vecs.size(0), self.input_size)

    # First fill in trees. Then, gather those added elements in a column, which become
    # the input to stack_rnn.
    if shift_batches.size(0) > 0:
      # this part should be one method? (to change behavior in a child class)
      shift_idx = stack.pointer[shift_batches].view(-1, 1, 1).expand(-1, 1, word_vecs.size(-1))
      shifted_embs = torch.gather(word_vecs[shift_batches], 1, shift_idx).squeeze(1)
      stack.do_shift(shift_batches, shifted_embs)
      new_input[shift_batches] = shifted_embs

    if nt_batches.size(0) > 0:
      nt_ids = (actions[nt_batches] - self.action_dict.nt_begin_id())
      nt_embs = self.nt_emb(nt_ids)
      stack.do_nt(nt_batches, nt_embs, nt_ids)
      new_input[nt_batches] = nt_embs

    if reduce_batches.size(0) > 0:
      children, ch_lengths, reduced_nt, reduced_nt_ids = stack.collect_reduced_children(reduce_batches)
      if isinstance(self.composition, AttentionComposition):
        hidden_head = stack.hiddens(batches=reduce_batches)[:, :, -1]
        stack_h = self.output(hidden_head)
      else:
        stack_h = None
      new_child, _, _ = self.composition(children, ch_lengths, reduced_nt, reduced_nt_ids, stack_h)
      stack.do_reduce(reduce_batches, new_child)
      new_input[reduce_batches] = new_child

    new_hidden, new_cell = self.stack_rnn(new_input, (stack.hidden_head(1), stack.cell_head(1)))

    stack.update_hidden(new_hidden, new_cell)
    # stack.update_nt_counts(actions, self.action_dict)

    ####
    #return self.output(stack.hidden_head()[:, :, -1])
    return stack.hidden_head()[:, :, -1]
    ####

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

    #### moved here for debug
    self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts(), w_dim), self.dropout)
    self.stack_rnn = MultiLayerLSTMCell(w_dim, h_dim, num_layers=num_layers, dropout=dropout)
    self.stack_to_hidden = nn.Sequential(self.dropout, nn.Linear(h_dim, w_dim), nn.ReLU())
    ####

    #self.rnng = RNNGCell(w_dim, h_dim, num_layers, dropout, self.action_dict, attention_composition)

    self.vocab_mlp = nn.Linear(w_dim, vocab)
    self.num_layers = num_layers
    self.num_actions = action_dict.num_actions()  # num_labels + 2
    self.action_mlp = nn.Linear(w_dim, self.num_actions)
    self.input_size = w_dim
    self.hidden_size = h_dim
    self.vocab_mlp.weight = self.emb[0].weight

    ###
    self.composition = (AttentionComposition(w_dim, dropout, self.action_dict.num_nts())
                        if attention_composition else
                        LSTMComposition(w_dim, dropout))
    ###

    self.max_open_nts = max_open_nts
    self.max_cons_nts = max_cons_nts

    ###
    self.initial_emb = nn.Sequential(nn.Embedding(1, w_dim), self.dropout)
    ###
    self.rnng = RNNGCell(w_dim, h_dim, num_layers, dropout, self.action_dict, attention_composition,
                         self.nt_emb, self.stack_rnn, self.stack_to_hidden, self.composition,
                         self.initial_emb)

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
    # hs = word_vecs.new_zeros(actions.size(1), word_vecs.size(0), self.input_size)
    # hs[0] = self.rnng.output(stack.hidden_head()[:, :, -1])
    hs = word_vecs.new_zeros(actions.size(1)+1, word_vecs.size(0), self.hidden_size)
    hs[0] = stack.hidden_head()[:, :, -1]
    for step in range(actions.size(1)):
      h = self.rnng(word_vecs, actions[:, step], stack)  # (batch_size, input_size)
      hs[step+1] = h
    hs = self.rnng.output(hs.transpose(1, 0)[:,:-1].contiguous())  # (batch_size, action_len, input_size)
    return hs

  def build_stack(self, x, batch_size = None):
    stack_size = max(150, x.size(1) + 100)
    batch_size = batch_size or x.size(0)
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

    beam = self.initial_beam(x)  # maybe want to calculate scores here.
    word_completed = [[] for _ in range(x.size(0))]
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    if return_beam_history:
      beam_histroy = []

    for pointer in range(x.size(1) + 1):
      forced_completions = [0 for _ in range(x.size(0))]
      bucket_i = 0
      while not all(len(batch_beam) == 0 for batch_beam in beam):
        new_beam, added_forced_completions = self.get_successors(
          x, pointer, beam, beam_size, shift_size)
        for i, c in enumerate(added_forced_completions):
          forced_completions[i] += c

        all_items, batch_idx = self._flatten_items(new_beam)

        self.update_stack_rnn_beam(all_items, word_vecs, batch_idx, pointer)

        beam_lengths = [len(batch_beam) for batch_beam in new_beam]
        self.update_beam_and_word_completed(
          beam, word_completed, all_items, beam_lengths, pointer == x.size(1))

        for b in range(len(beam)):
          if len(word_completed[b]) - forced_completions[b] >= beam_size:
            beam[b] = []
        if return_beam_history:
          beam_histroy.append((pointer,
                               bucket_i,
                               [b[:] for b in beam],
                               [b[:] for b in word_completed]))
        bucket_i += 1

      for b in range(len(beam)):
        word_completed[b].sort(key=lambda x: x.score, reverse=True)
        beam[b] = word_completed[b][:word_beam_size]
        word_completed[b] = []

      marginal = self._get_marginal_ll(beam)
      for b, s in enumerate(marginal):
        word_marginal_ll[b].append(s)

    parses = [[(item.parse_actions(), item.score) for item in  batch_beam]
              for batch_beam in beam]
    surprisals = [[] for _ in range(len(beam))]
    for b in range(len(beam)):
      for i in range(0, len(word_marginal_ll[b])-1):
        surprisals[b].append(-word_marginal_ll[b][i] - (-word_marginal_ll[b][i-1] if i > 0 else 0))

    ret = (parses, surprisals)
    if return_beam_history:
      ret += (beam_histroy,)
    return ret

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
    initial_hidden = self.rnng.get_initial_hidden(x)  # [(batch_size, hidden_size), (batch_size, hidden_size)]
    return [TopDownState.from_initial_stack((initial_hidden[0][b], initial_hidden[1][b]))
            for b in range(x.size(0))]

  def initial_particle_beam(self, x, K):
    states = self.initial_states(x)
    return [[ParticleBeamItem.from_initial_state(state, K)] for state in states]

  def get_successors(self, x, pointer, beam, beam_size, shift_size):
    all_items, _ = self._flatten_items(beam)
    beam_lengths = [len(batch_beam) for batch_beam in beam]
    states = [item.state for item in all_items]

    if pointer < x.size(1):
      next_x = [x[b, pointer].expand(beam_lengths[b]) for b in range(x.size(0))]
      next_x = torch.cat(next_x)  # (total beam size)
    else:
      next_x = None

    action_mask = self.valid_action_mask(all_items, x.size(1))  # (total beam size, n_actions)

    log_probs = self.action_log_probs(states, action_mask, next_x)  # tensor of size (total beam, n_actions)
    log_probs += log_probs.new_tensor([item.score for item in all_items]).unsqueeze(1)
    log_probs = self._deflatten_items(log_probs, beam_lengths)
    successors, forced_completions = self.scores_to_successors(
      x, pointer, beam, log_probs, beam_size, shift_size)

    successors = [[beam[batch_i][b].next_incomplete_item(a, s) for (b, a, s) in batch_succs]
                  for batch_i, batch_succs in enumerate(successors)]

    return successors, forced_completions

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

  def action_log_probs(self, states, action_mask, next_x = None, return_disc_probs = False):
    """
    states: tensor of size (batch, hidden_size)
    new_x: tensor of size (batch)
    """
    hiddens = self.stack_top_h(states)  # (total beam size, hidden_size)
    hiddens = self.rnng.output(hiddens)  # (total beam size, hidden_size)

    action_logit = self.action_mlp(hiddens)
    action_logit[action_mask != 1] = -float('inf')

    log_probs = F.log_softmax(action_logit, -1)  # (total beam size, num_actions)
    if return_disc_probs:
      disc_log_probs = log_probs.clone()

    if next_x is not None:  # shift is valid for next action
      word_logit = self.vocab_mlp(hiddens)
      shift_idx = self.action_dict.a2i['SHIFT']
      assert next_x.size(0) == hiddens.size(0)
      word_log_probs = self.word_criterion(word_logit, next_x) * -1.0  # (total beam size)
      log_probs[:, shift_idx] += word_log_probs

    if return_disc_probs:
      return (log_probs, disc_log_probs)
    else:
      return log_probs

  def valid_action_mask(self, items, sent_len):
    mask = torch.ones((len(items), self.num_actions), dtype=torch.uint8)
    mask[:, self.action_dict.padding_idx] = 0
    for b, item in enumerate(items):
      state = item.state
      prev_action = item.action
      if state.pointer == sent_len and state.nopen_parens == 0:  # finished
        mask[b, :] = 0
        continue
      if state.nopen_parens == 0:  # only nt
        assert state.pointer == 0 and len(state.hiddens) == 1
        self.action_dict.mask_shift(mask, b)
        self.action_dict.mask_reduce(mask, b)
      if state.pointer == sent_len:  # only reduce
        self.action_dict.mask_nt(mask, b)
        self.action_dict.mask_shift(mask, b)
      if (state.nopen_parens > self.max_open_nts or
          state.ncons_nts >= self.max_cons_nts):  # no more open
        self.action_dict.mask_nt(mask, b)

      if ((state.nopen_parens == 1 and state.pointer < sent_len) or # cannot reduce the unique open element at intermediate steps.
          self.action_dict.is_nt(prev_action) or # cannot reduce immediately after nt
          len(state.hiddens) < 3):
        self.action_dict.mask_reduce(mask, b)
    mask = mask.to(items[0].state.hiddens[0].device)

    return mask

  def stack_top_h(self, states):
    return torch.stack([state.hiddens[-1][:, -1] for state in states], dim=0)

  def scores_to_successors(self, x, pointer, beam, total_scores, beam_size, shift_size):
    successors = [[] for _ in range(len(total_scores))]
    forced_completions = [0 for _ in range(len(total_scores))]
    for batch, batch_total_scores in enumerate(total_scores):
      if batch_total_scores.size(0) == 0:
        continue  # this batch is finished.
      _, num_actions = batch_total_scores.size()

      scores = batch_total_scores.view(-1)
      sorted_scores, sort_idx = torch.sort(scores, descending=True)

      active_span = (sorted_scores != -float('inf')).nonzero()[-1][0] + 1
      sorted_scores = sorted_scores[:active_span]
      sort_idx = sort_idx[:active_span]

      beam_id = sort_idx // num_actions
      action_id = sort_idx % num_actions

      beam_id_np = beam_id.cpu().numpy()
      action_id_np = action_id.cpu().numpy()
      sorted_scores_np = sorted_scores.cpu().numpy()

      successors[batch] = [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                           for i in range(min(len(beam_id_np), beam_size))]
      if pointer < x.size(1):
        # shift is target for being forced.
        shift_in_successors = (action_id[:beam_size] == self.action_dict.a2i['SHIFT']).nonzero().size(0)
        if shift_in_successors < shift_size:
          # find and add additional forced shift successors
          additional = ((action_id[beam_size:] == self.action_dict.a2i['SHIFT'])
                        .nonzero()[:shift_size - shift_in_successors].squeeze(1)).cpu()
          additional += beam_size  # add offset
          assert all(action_id_np[i] == self.action_dict.a2i['SHIFT'] for i in additional)
          successors[batch] += [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                                for i in additional]
          forced_completions[batch] = additional.size(0)
      else:
        # At the end, save final actions.
        finish_actions = [(b, a, s) for (b, a, s) in successors[batch]
                          if self._is_last_action(a, beam[batch][b].state, True)]

        if len(finish_actions) < shift_size:
          remain = shift_size - len(finish_actions)
          additional = []
          for i in range(beam_size, len(beam_id_np)):
            b = beam_id_np[i]
            a = action_id_np[i]
            if self._is_last_action(a, beam[batch][b].state, True):
              additional.append(i)
            if len(additional) >= remain:
              break
          successors[batch] += [(beam_id_np[i], action_id_np[i], sorted_scores_np[i])
                                for i in additional]
          forced_completions[batch] = len(additional)

    return successors, forced_completions

  def update_stack_rnn_beam(self, items, word_vecs, batch_idx, pointer):
    """
    This method does some preparation for update_stack_rnn.

    items is a flattened tensor. batch_idx is mapping from index in flattend
    list to the original batch, which is necessary to get correct mapping from
    word_vecs, which is not flattened and has size of (batch_size, num words).
    """
    states = [item.state for item in items]
    actions = word_vecs.new_tensor([item.action for item in items], dtype=torch.long)

    batch_idx_tensor = word_vecs.new_tensor(batch_idx, dtype=torch.long)
    shifts = (actions == self.action_dict.a2i['SHIFT']).nonzero().squeeze(1)

    shifted_batch_idx = batch_idx_tensor[shifts]
    if pointer < word_vecs.size(1):
      shifted_words = word_vecs[shifted_batch_idx, pointer]
    else:
      # At end of sentence. Since shift is not allowed, shifted_words will become
      # empty. We have to process this condition separately.
      assert shifted_batch_idx.size(0) == 0
      shifted_words = word_vecs[shifted_batch_idx, -1]

    self.update_stack_rnn(states, actions, shifts, shifted_words)

  def update_stack_rnn(self, states, actions, shift_idx, shifted_embs):
    assert actions.size(0) == len(states)
    assert shift_idx.size(0) == shifted_embs.size(0)
    if len(states) == 0:
      return

    reduces = (actions == self.action_dict.a2i['REDUCE']).nonzero().squeeze(1)
    nts = (actions >= self.action_dict.nt_begin_id()).nonzero().squeeze(1)

    new_stack_input = shifted_embs.new_zeros(actions.size(0), self.input_size)

    if reduces.size(0) > 0:
      reduce_idx = reduces.cpu().numpy()
      reduce_states = [states[i] for i in reduce_idx]
      children, ch_lengths, nt, nt_id = self._collect_children_for_reduce(reduce_states)
      if isinstance(self.rnng.composition, AttentionComposition):
        reduce_context = self.stack_top_h(reduce_states)
        reduce_context = self.rnng.output(reduce_context)
      else:
        reduce_context = None
      new_child, _, _ = self.rnng.composition(children, ch_lengths, nt, nt_id, reduce_context)
      new_stack_input[reduces] = new_child.float()

    new_stack_input[shift_idx] = shifted_embs

    nt_ids = (actions[nts] - self.action_dict.nt_begin_id())
    nt_embs = self.rnng.nt_emb(nt_ids)
    new_stack_input[nts] = nt_embs

    stack_top_context = self._collect_stack_top_context(states)
    new_stack_top = self.rnng.stack_rnn(new_stack_input, stack_top_context)

    hiddens, cells = new_stack_top
    for b in range(len(states)):
      a = actions[b].item()
      states[b].update_stack((hiddens[b], cells[b]), new_stack_input[b], a, self.action_dict)
      states[b].do_action(a, self.action_dict)

    #return self.rnng.output(new_stack_top[:, :, -1])

  def update_beam_and_word_completed(self, beam, word_completed, items, beam_lengths, last_token=False):
    accum = 0
    for b in range(len(beam)):
      beam[b] = []
      l = beam_lengths[b]
      for item in items[accum:accum+l]:
        if (item.state.finished() or self.action_dict.is_shift(item.action)):
          # Cases where shifting last+1 token should be pruned by action masking.
          word_completed[b].append(item)
        else:
          beam[b].append(item)
      accum += l
    assert accum == sum(beam_lengths)

  def _is_last_action(self, action, state, shifted_all):
    return (shifted_all and self.action_dict.is_reduce(action) and
            state.can_finish_by_reduce())

  def _flatten_items(self, items):
    flatten_items = []
    idxs = []
    for i, batch_items in enumerate(items):
      flatten_items += batch_items
      idxs += [i] * len(batch_items)
    return flatten_items, idxs

  def _deflatten_items(self, items, beam_lengths):
    de_items = []
    accum_idx = 0
    for b, b_len in enumerate(beam_lengths):
      de_items.append(items[accum_idx:accum_idx+b_len])
      accum_idx += b_len
    assert accum_idx == len(items)
    return de_items  # List[Tensor]

  def _collect_children_for_reduce(self, reduce_states):
    children = []
    nt = []
    nt_ids = []
    ch_lengths = []
    for state in reduce_states:
      reduced_nt, reduced_trees, nt_id = state.reduce_stack()
      nt.append(reduced_nt)
      nt_ids.append(nt_id)
      children.append(reduced_trees)
      ch_lengths.append(len(reduced_trees))

    max_ch_len = max(ch_lengths)
    zero_child = children[0][0].new_zeros(children[0][0].size())
    for b, b_child in enumerate(children):
      if len(b_child) < max_ch_len:
        children[b] += [zero_child] * (max_ch_len - len(b_child))
      children[b] = torch.stack(children[b], 0)

    children = torch.stack(children, 0)
    nt = torch.stack(nt, 0)
    nt_ids = nt.new_tensor(nt_ids, dtype=torch.long)
    ch_lengths = nt_ids.new_tensor(ch_lengths)

    return (children, ch_lengths, nt, nt_ids)

  def _collect_stack_top_context(self, states, idx = None, offset = 1):
    if idx is None:
      idx = range(len(states))
    stack_h_all = []
    hs = torch.stack([states[i].hiddens[-offset] for i in idx])
    cs = torch.stack([states[i].cells[-offset] for i in idx])
    return hs, cs

  def _get_marginal_ll(self, beam):
    ll = []
    for b in range(len(beam)):
      scores = [item.score for item in beam[b]]
      ll.append(torch.logsumexp(torch.tensor(scores), 0).item())
    return ll


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
