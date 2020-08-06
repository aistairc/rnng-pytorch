import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils import *
from TreeCRF import ConstituencyTreeCRF
from torch.distributions import Bernoulli
import itertools

class RNNLM(nn.Module):
  def __init__(self, vocab=10000,
               w_dim=650,
               h_dim=650,
               num_layers=2,
               dropout=0.5):
    super(RNNLM, self).__init__()
    self.h_dim = h_dim
    self.num_layers = num_layers    
    self.word_vecs = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)
    self.rnn = nn.LSTM(w_dim, h_dim, num_layers = num_layers,
                       dropout = dropout, batch_first = True)      
    self.vocab_linear =  nn.Linear(h_dim, vocab)
    self.vocab_linear.weight = self.word_vecs.weight # weight sharing

  def forward(self, sent):
    word_vecs = self.dropout(self.word_vecs(sent[:, :-1]))
    h, _ = self.rnn(word_vecs)
    log_prob = F.log_softmax(self.vocab_linear(self.dropout(h)), 2) # b x l x v
    ll = torch.gather(log_prob, 2, sent[:, 1:].unsqueeze(2)).squeeze(2)
    return ll.sum(1)
  
  def generate(self, bos = 2, eos = 3, max_len = 150):
    x = []
    bos = torch.LongTensor(1,1).cuda().fill_(bos)
    emb = self.dropout(self.word_vecs(bos))
    prev_h = None
    for l in range(max_len):
      h, prev_h = self.rnn(emb, prev_h)
      prob = F.softmax(self.vocab_linear(self.dropout(h.squeeze(1))), 1)
      sample = torch.multinomial(prob, 1)
      emb = self.dropout(self.word_vecs(sample))
      x.append(sample.item())
      if x[-1] == eos:
        x.pop()
        break
    return x

class SeqLSTM(nn.Module):
  def __init__(self, i_dim = 200,  # word embedding dim
               h_dim = 0,  # internal hidden dim
               num_layers = 1,
               dropout = 0):
    super(SeqLSTM, self).__init__()    
    self.i_dim = i_dim
    self.h_dim = h_dim
    self.num_layers = num_layers
    self.linears = nn.ModuleList([nn.Linear(h_dim + i_dim, h_dim*4) if l == 0 else
                                  nn.Linear(h_dim*2, h_dim*4) for l in range(num_layers)])
    self.dropout = dropout
    self.dropout_layer = nn.Dropout(dropout)

  def forward(self, x, prev_h = None):
    if prev_h is None:
      prev_h = [(x.new(x.size(0), self.h_dim).fill_(0),
                 x.new(x.size(0), self.h_dim).fill_(0)) for _ in range(self.num_layers)]  # (n_layers, 2, batch_size, h_dim)
    curr_h = []
    for l in range(self.num_layers):
      input = x if l == 0 else curr_h[l-1][0]
      if l > 0 and self.dropout > 0:
        input = self.dropout_layer(input)
      concat = torch.cat([input, prev_h[l][0]], 1)  # (batch_size, h_dim*2) or (batch_size, h_dim+i_dim)
      all_sum = self.linears[l](concat)  # (batch_size, h_dim*4)
      i, f, o, g = all_sum.split(self.h_dim, 1)  # (batch_size, h_dim) for each
      c = torch.sigmoid(f)*prev_h[l][1] + torch.sigmoid(i)*torch.tanh(g)
      h = torch.sigmoid(o)*torch.tanh(c)
      curr_h.append((h, c))
    return curr_h  # (n_layers, 2, batch_size, h_dim)

class TreeLSTM(nn.Module):
  def __init__(self, dim = 200):
    super(TreeLSTM, self).__init__()
    self.dim = dim
    self.linear = nn.Linear(dim*2, dim*5)

  def forward(self, x1, x2, e=None):
    if not isinstance(x1, tuple):
      x1 = (x1, None)    
    h1, c1 = x1 
    if x2 is None: 
      x2 = (torch.zeros_like(h1), torch.zeros_like(h1))
    elif not isinstance(x2, tuple):
      x2 = (x2, None)    
    h2, c2 = x2
    if c1 is None:
      c1 = torch.zeros_like(h1)
    if c2 is None:
      c2 = torch.zeros_like(h2)
    concat = torch.cat([h1, h2], 1)
    all_sum = self.linear(concat)
    i, f1, f2, o, g = all_sum.split(self.dim, 1)

    c = torch.sigmoid(f1)*c1 + torch.sigmoid(f2)*c2 + torch.sigmoid(i)*torch.tanh(g)
    h = torch.sigmoid(o)*torch.tanh(c)
    return (h, c)

class RNNG(nn.Module):
  def __init__(self, vocab = 100,
               w_dim = 20, 
               h_dim = 20,
               num_layers = 1,
               dropout = 0,
               q_dim = 20,
               max_len = 250):
    super(RNNG, self).__init__()
    self.S = 0 #action idx for shift/generate
    self.R = 1 #action idx for reduce
    self.emb = nn.Embedding(vocab, w_dim)
    self.dropout = nn.Dropout(dropout)    
    self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers = num_layers, dropout = dropout)
    self.tree_rnn = TreeLSTM(w_dim)
    self.vocab_mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, vocab))
    self.num_layers = num_layers
    self.q_binary = nn.Sequential(nn.Linear(q_dim*2, q_dim*2), nn.ReLU(), nn.LayerNorm(q_dim*2),
                                  nn.Dropout(dropout), nn.Linear(q_dim*2, 1))
    self.action_mlp_p = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, 1))
    self.w_dim = w_dim
    self.h_dim = h_dim
    self.q_dim = q_dim    
    self.q_leaf_rnn = nn.LSTM(w_dim, q_dim, bidirectional = True, batch_first = True)
    self.q_crf = ConstituencyTreeCRF()
    self.pad1 = 0 # idx for <s> token from ptb.dict
    self.pad2 = 2 # idx for </s> token from ptb.dict 
    self.q_pos_emb = nn.Embedding(max_len, w_dim) # position embeddings
    self.vocab_mlp[-1].weight = self.emb.weight #share embeddings

  def get_span_scores(self, x):
    #produces the span scores s_ij
    bos = x.new(x.size(0), 1).fill_(self.pad1)
    eos  = x.new(x.size(0), 1).fill_(self.pad2)
    x = torch.cat([bos, x, eos], 1)  # (batch_size, n_words+2)
    x_vec = self.dropout(self.emb(x))
    pos = torch.arange(0, x.size(1)).unsqueeze(0).expand_as(x).long().cuda()  # (batch_size, n_words)
    x_vec = x_vec + self.dropout(self.q_pos_emb(pos))
    q_h, _ = self.q_leaf_rnn(x_vec)
    fwd = q_h[:, 1:, :self.q_dim]  # (batch_size, n_words+1, q_dim)
    bwd = q_h[:, :-1, self.q_dim:]  # (batch_size, n_words+1, q_dim)
    fwd_diff = fwd[:, 1:].unsqueeze(1) - fwd[:, :-1].unsqueeze(2)
    bwd_diff = bwd[:, :-1].unsqueeze(2) - bwd[:, 1:].unsqueeze(1)
    concat = torch.cat([fwd_diff, bwd_diff], 3)
    scores = self.q_binary(concat).squeeze(3)
    return scores  # (batch_size, n_words, n_words)

  def get_action_masks(self, actions, length):
    #this masks out actions so that we don't incur a loss if some actions are deterministic
    #in practice this doesn't really seem to matter
    mask = actions.new(actions.size(0), actions.size(1)).fill_(1)
    for b in range(actions.size(0)):      
      num_shift = 0
      stack_len = 0
      for l in range(actions.size(1)):
        if stack_len < 2:
          mask[b][l].fill_(0)
        if actions[b][l].item() == self.S:
          num_shift += 1
          stack_len += 1
        else:
          stack_len -= 1
    return mask

  def forward(self, x, samples = 1, is_temp = 1., has_eos=True):
    #For has eos, if </s> exists, then inference network ignores it. 
    #Note that </s> is predicted for training since we want the model to know when to stop.
    #However it is ignored for PPL evaluation on the version of the PTB dataset from
    #the original RNNG paper (Dyer et al. 2016)
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]
    batch, length = x.size(0), x.size(1)
    if has_eos: 
      parse_length = length - 1
      parse_x = x[:, :-1]
    else:
      parse_length = length
      parse_x = x
    word_vecs =  self.dropout(self.emb(x))
    scores = self.get_span_scores(parse_x)
    self.scores = scores
    scores = scores / is_temp
    self.q_crf._forward(scores)
    self.q_crf._entropy(scores)
    entropy = self.q_crf.entropy[0][parse_length-1]
    crf_input = scores.unsqueeze(1).expand(batch, samples, parse_length, parse_length)
    crf_input = crf_input.contiguous().view(batch*samples, parse_length, parse_length)
    for i in range(len(self.q_crf.alpha)):
      for j in range(len(self.q_crf.alpha)):
        self.q_crf.alpha[i][j] = self.q_crf.alpha[i][j].unsqueeze(1).expand(
          batch, samples).contiguous().view(batch*samples)        
    _, log_probs_action_q, tree_brackets, spans = self.q_crf._sample(crf_input, self.q_crf.alpha)
    actions = []
    for b in range(crf_input.size(0)):    
      action = get_actions(tree_brackets[b])
      if has_eos:
        actions.append(action + [self.S, self.R]) #we train the model to generate <s> and then do a final reduce
      else:
        actions.append(action)
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length) 
    num_action = 2*length - 1
    batch_expand = batch*samples
    contexts = []
    log_probs_action_p = [] #conditional prior
    init_emb = init_emb.unsqueeze(1).expand(batch, samples, self.w_dim)
    init_emb = init_emb.contiguous().view(batch_expand, self.w_dim)
    init_stack = self.stack_rnn(init_emb, None)
    x_expand = x.unsqueeze(1).expand(batch, samples, length)
    x_expand = x_expand.contiguous().view(batch_expand, length)
    word_vecs = self.dropout(self.emb(x_expand))
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]
    stack_child = [[] for _ in range(batch_expand)]
    stack2 = [[] for _ in range(batch_expand)]
    for b in range(batch_expand):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    pointer = [0]*batch_expand
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch_expand):
        # batch all the shift/reduce operations separately
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))

      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]
      for b in range(batch_expand):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)
      stack.append(stack_h)
      for b in range(batch_expand):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
      
    contexts = torch.stack(contexts, 1) #stack contexts
    action_logit_p = self.action_mlp_p(contexts).squeeze(2) 
    action_prob_p = torch.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch*samples, length, self.h_dim)

    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x_expand.unsqueeze(2)).squeeze(2)
    log_probs_word = log_probs_word.sum(1)
    log_probs_word = log_probs_word.contiguous().view(batch, samples)
    log_probs_action_p = action_score.contiguous().view(batch, samples)
    log_probs_action_q = log_probs_action_q.contiguous().view(batch, samples)
    actions = actions.contiguous().view(batch, samples, -1)
    return log_probs_word, log_probs_action_p, log_probs_action_q, actions, entropy

  def forward_actions(self, x, actions, has_eos=True):
    # this is for when ground through actions are available
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:]    
    if has_eos:
      new_actions = []
      for action in actions:
        new_actions.append(action + [self.S, self.R])
      actions = new_actions
    batch, length = x.size(0), x.size(1)
    word_vecs =  self.dropout(self.emb(x))
    actions = torch.Tensor(actions).float().cuda()
    action_masks = self.get_action_masks(actions, length)  # for not incurring loss when next action is deterministic (shift is deterministic when stack size <= 1)
    num_action = 2*length - 1
    contexts = []
    log_probs_action_p = [] #prior
    init_stack = self.stack_rnn(init_emb, None)
    word_vecs = word_vecs.unsqueeze(2)
    word_vecs_zeros = torch.zeros_like(word_vecs)
    stack = [init_stack]  # keeps each step's top stack element
    stack_child = [[] for _ in range(batch)]  # batch to current stack elements (used for TreeLSTM)
    stack2 = [[] for _ in range(batch)]  # batch to current stack elements (used for stack LSTM)
    pointer = [0]*batch
    for b in range(batch):
      stack2[b].append([[init_stack[l][0][b], init_stack[l][1][b]] for l in range(self.num_layers)])
    for l in range(num_action):
      contexts.append(stack[-1][-1][0])  # stack top state (hidden)
      stack_input = []
      child1_h = []
      child1_c = []
      child2_h = []
      child2_c = []
      stack_context = []
      for b in range(batch):
        if actions[b][l].item() == self.R:
          child1 = stack_child[b].pop()
          child2 = stack_child[b].pop()
          child1_h.append(child1[0])
          child1_c.append(child1[1])
          child2_h.append(child2[0])
          child2_c.append(child2[1])
          stack2[b].pop()
          stack2[b].pop()
      if len(child1_h) > 0:
        child1_h = torch.cat(child1_h, 0)
        child1_c = torch.cat(child1_c, 0)
        child2_h = torch.cat(child2_h, 0)
        child2_c = torch.cat(child2_c, 0)
        new_child = self.tree_rnn((child1_h, child1_c), (child2_h, child2_c))
      child_idx = 0
      stack_h = [[[], []] for _ in range(self.num_layers)]  # (n_layers, 2, batch_size, h_dim); last layer of stack2
      for b in range(batch):
        assert(len(stack2[b]) - 1 == len(stack_child[b]))
        for k in range(self.num_layers):
          stack_h[k][0].append(stack2[b][-1][k][0])
          stack_h[k][1].append(stack2[b][-1][k][1])
        if actions[b][l].item() == self.S:          
          input_b = word_vecs[b][pointer[b]]
          stack_child[b].append((word_vecs[b][pointer[b]], word_vecs_zeros[b][pointer[b]]))  # correspond to (h, c)
          pointer[b] += 1          
        else:
          input_b = new_child[0][child_idx].unsqueeze(0)
          stack_child[b].append((input_b, new_child[1][child_idx].unsqueeze(0)))
          child_idx += 1
        stack_input.append(input_b)
      stack_input = torch.cat(stack_input, 0)
      stack_h_all = []
      for k in range(self.num_layers):
        stack_h_all.append((torch.stack(stack_h[k][0], 0), torch.stack(stack_h[k][1], 0)))
      stack_h = self.stack_rnn(stack_input, stack_h_all)  # Each step always update the top element of stack. This updates the stack with that new element.
      stack.append(stack_h)
      for b in range(batch):
        stack2[b].append([[stack_h[k][0][b], stack_h[k][1][b]] for k in range(self.num_layers)])
    contexts = torch.stack(contexts, 1)  # (batch_size, n_actions, h_dim)
    action_logit_p = self.action_mlp_p(contexts).squeeze(2)
    action_prob_p = torch.sigmoid(action_logit_p).clamp(min=1e-7, max=1-1e-7)
    action_shift_score = (1 - action_prob_p).log()
    action_reduce_score = action_prob_p.log()
    action_score = (1-actions)*action_shift_score + actions*action_reduce_score
    action_score = (action_score*action_masks).sum(1)
    
    word_contexts = contexts[actions < 1]
    word_contexts = word_contexts.contiguous().view(batch, length, self.h_dim)
    log_probs_word = F.log_softmax(self.vocab_mlp(word_contexts), 2)
    log_probs_word = torch.gather(log_probs_word, 2, x.unsqueeze(2)).squeeze(2).sum(1)
    log_probs_action_p = action_score.contiguous().view(batch)
    actions = actions.contiguous().view(batch, 1, -1)
    return log_probs_word, log_probs_action_p, actions
  
  def forward_tree(self, x, actions, has_eos=True):
    # this is log q( tree | x) for discriminative parser training in supervised RNNG
    init_emb = self.dropout(self.emb(x[:, 0]))
    x = x[:, 1:-1]
    batch, length = x.size(0), x.size(1)
    scores = self.get_span_scores(x)
    crf_input = scores
    gold_spans = scores.new(batch, length, length)
    for b in range(batch):
      gold_spans[b].copy_(torch.eye(length).cuda())  # eye sets up token (single word) spans
      spans = get_spans(actions[b])
      for span in spans:
        gold_spans[b][span[0]][span[1]] = 1
    self.q_crf._forward(crf_input)
    log_Z = self.q_crf.alpha[0][length-1]
    span_scores = (gold_spans*scores).sum(2).sum(1)
    ll_action_q = span_scores - log_Z
    return ll_action_q
    
  def logsumexp(self, x, dim=1):
    d = torch.max(x, dim)[0]    
    if x.dim() == 1:
      return torch.log(torch.exp(x - d).sum(dim)) + d
    else:
      return torch.log(torch.exp(x - d.unsqueeze(dim).expand_as(x)).sum(dim)) + d    

class LSTMComposition(nn.Module):
  def __init__(self, dim, dropout):
    super(LSTMComposition, self).__init__()
    self.dim = dim
    self.rnn = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
    self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim*2, dim), nn.ReLU())

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):
    lengths = ch_lengths + 2
    nt = nt.unsqueeze(1)
    elems = torch.cat([nt, children, torch.zeros_like(nt)], dim=1)
    for b, l in enumerate(lengths):
      elems[b][l-1] = nt[b]

    packed = pack_padded_sequence(elems, lengths, batch_first=True, enforce_sorted=False)
    h, _ = self.rnn(packed)
    h, _ = pad_packed_sequence(h, batch_first=True)

    gather_idx = (lengths - 2).unsqueeze(1).expand(-1, h.size(-1)).unsqueeze(1)
    fwd = h.gather(1, gather_idx).squeeze(1)[:, :self.dim]
    bwd = h[:, 1, self.dim:]
    c = torch.cat([fwd, bwd], dim=1)

    return self.output(c), None, None

class AttentionComposition(nn.Module):
  def __init__(self, w_dim, h_dim, num_labels = 10):
    super(AttentionComposition, self).__init__()
    self.w_dim = w_dim
    self.h_dim = h_dim
    self.num_labels = num_labels

    # self.V = nn.Bilinear(w_dim, h_dim+w_dim, )
    self.V = nn.Linear(w_dim, h_dim+w_dim, bias=False)
    self.nt_emb = nn.Embedding(num_labels, w_dim)  # o_nt in the Kuncoro et al. (2017)
    self.nt_emb2 = nn.Embedding(num_labels, w_dim)  # t_nt in the Kuncoro et al. (2017)
    self.gate = nn.Sequential(nn.Linear(w_dim*2, w_dim), nn.Sigmoid())

  def forward(self, children, ch_lengths, nt, nt_id, stack_state):  # children: (batch_size, n_children, w_dim)
    rhs = torch.cat([self.nt_emb(nt_id), stack_state], dim=1) # (batch_size, h_dim+w_dim, 1)
    logit = (self.V(children)*rhs.unsqueeze(1)).sum(-1)  # equivalent to bmm(self.V(children), rhs.unsqueeze(-1)).squeeze(-1)
    len_mask = length_to_mask(ch_lengths)
    logit[len_mask != 1] = -float('inf')
    attn = F.softmax(logit)
    weighted_child = (children*attn.unsqueeze(-1)).sum(1)

    nt2 = self.nt_emb2(nt_id)  # (batch_size, w_dim)
    gate_input = torch.cat([nt2, weighted_child], dim=-1)
    g = self.gate(gate_input)  # (batch_size, w_dim)
    c = g*nt2 + (1-g)*weighted_child  # (batch_size, w_dim)

    return c, attn, g

# class AbstractRNNG(nn.Module):
#   def __init__(self):
#     super(AbstractRNNG, self).__init__()

class TopDownRNNG(nn.Module):
  def __init__(self, action_dict,  # need to define a class
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
    super(TopDownRNNG, self).__init__()
    self.action_dict = action_dict
    self.padding_idx = padding_idx
    self.action_criterion = nn.CrossEntropyLoss(reduction='none',
                                                ignore_index=action_dict.padding_idx)
    self.word_criterion = nn.CrossEntropyLoss(reduction='none',
                                              ignore_index=padding_idx)

    self.dropout = nn.Dropout(dropout)
    self.emb = nn.Sequential(
      nn.Embedding(vocab, w_dim, padding_idx=padding_idx), self.dropout)
    self.nt_emb = nn.Sequential(nn.Embedding(action_dict.num_nts(), w_dim), self.dropout)
    self.stack_rnn = SeqLSTM(w_dim, h_dim, num_layers=num_layers, dropout=dropout)
    self.stack_to_hidden = nn.Sequential(self.dropout, nn.Linear(h_dim, h_dim), nn.ReLU())
    self.vocab_mlp = nn.Linear(h_dim, vocab)
    self.num_layers = num_layers
    self.num_actions = action_dict.num_actions()  # num_labels + 2
    self.action_mlp = nn.Linear(h_dim, self.num_actions)
    self.w_dim = w_dim
    self.h_dim = h_dim
    if w_dim == h_dim:
      self.vocab_mlp.weight = self.emb[0].weight

    self.composition = (AttentionComposition(w_dim, h_dim, self.action_dict.num_nts())
                        if attention_composition else
                        LSTMComposition(w_dim, dropout))
    self.max_open_nts = max_open_nts
    self.max_cons_nts = max_cons_nts

    self.initial_emb = nn.Sequential(nn.Embedding(1, w_dim), self.dropout)

    # self.zero_word = self.initial_emb[0].weight.new_zeros(w_dim)

  def forward(self, x, actions, initial_stack = None):
    assert isinstance(x, torch.Tensor)
    assert isinstance(actions, torch.Tensor)
    if initial_stack is None:
      initial_stack = self.get_initial_stack(x)
    state = TopDownState(initial_stack, x.size(1))

    word_vecs = self.emb(x)
    step = self.unroll_state(state, word_vecs, actions)
    assert step == actions.size(1)

    action_contexts = state.action_contexts()
    action_contexts = self.stack_to_hidden(action_contexts)
    a_loss, _ = self.action_loss(actions, self.action_dict, action_contexts)
    w_loss, _ = self.word_loss(x, actions, self.action_dict, action_contexts)
    # loss = (a_loss.sum() + w_loss.sum()) / (a_loss.size(0) + w_loss.size(0))
    loss = (a_loss.sum() + w_loss.sum()) / a_loss.size(0)
    return loss, a_loss, w_loss, state

  def unroll_state(self, state, word_vecs, actions):
    step = 0
    while not state.all_finished():
      assert step < actions.size(1)
      self.one_step_for_train(state, word_vecs, actions[:, step])
      step += 1
    return step

  def one_step_for_train(self, state, word_vecs, action):
    reduce_batch = self._reduce_batch(state, action)
    non_reduce_batch = self._non_reduce_batch(state, action)
    if len(reduce_batch) > 0:
      children, ch_lengths, nt, nt_id = self._collect_children_for_reduce(state, reduce_batch)
      reduce_context = state.stack_top_h(reduce_batch)
      reduce_context = self.stack_to_hidden(reduce_context)
      new_child, _, _ = self.composition(children, ch_lengths, nt, nt_id, reduce_context)
    else:
      new_child = None

    new_stack_input = [None] * action.size(0)
    for i, b in enumerate(reduce_batch):
      new_stack_input[b] = new_child[i]
    for b in non_reduce_batch:
      a = action[b]
      if self.action_dict.is_shift(a):
        new_stack_input[b] = word_vecs[b, state.pointer[b]]
      elif self.action_dict.is_pad(a):  # finished state
        new_stack_input[b] = self.emb(word_vecs.new_tensor([self.padding_idx],
                                                           dtype=torch.long)).squeeze()
      else:
        assert self.action_dict.is_nt(a)
        nt_id = self.action_dict.nt_id(a)
        new_stack_input[b] = self.nt_emb(word_vecs.new_tensor([nt_id], dtype=torch.long)).squeeze()
    new_stack_input = torch.stack(new_stack_input, 0)
    stack_top_context = state.stack_top_context()
    new_stack_top = self.stack_rnn(new_stack_input, stack_top_context)
    state.update_stack(new_stack_top, new_stack_input)
    state.do_action(action, self.action_dict)

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

  def get_initial_stack(self, x):
    iemb = self.initial_emb(x.new_zeros(x.size(0)))
    return self.stack_rnn(iemb, None)

  def _reduce_batch(self, state, action):
    return [b for b, a in enumerate(action) if self.action_dict.is_reduce(a)]

  def _non_reduce_batch(self, state, action):
    return [b for b, a in enumerate(action) if not self.action_dict.is_reduce(a)]

  def _collect_children_for_reduce(self, state, reduce_batch):
    children = []
    nt = []
    nt_ids = []
    ch_lengths = []
    for b in reduce_batch:
      reduced_nt, reduced_trees, nt_id = state.reduce_stack(b)
      nt.append(reduced_nt)
      nt_ids.append(nt_id)
      children.append(reduced_trees)
      ch_lengths.append(len(reduced_trees))

    max_ch_len = max(ch_lengths)
    for b, b_child in enumerate(children):
      if len(b_child) < max_ch_len:
        children[b] += [b_child[0].new_zeros(b_child[0].size())] * (max_ch_len - len(b_child))
      children[b] = torch.stack(children[b], 0)

    children = torch.stack(children, 0)
    nt = torch.stack(nt, 0)
    nt_ids = nt.new_tensor(nt_ids, dtype=torch.long)
    ch_lengths = nt_ids.new_tensor(ch_lengths)

    return (children, ch_lengths, nt, nt_ids)

  def word_sync_beam_search(self, x, beam_size, word_beam_size = 0, shift_size = 0):
    self.eval()

    if word_beam_size <= 0:
      word_beam_size = beam_size

    beam = self.initial_beam(x)
    word_completed = [[] for _ in range(x.size(0))]
    word_vecs = self.emb(x)
    word_marginal_ll = [[] for _ in range(x.size(0))]

    for pointer in range(x.size(1) + 1):
      forced_completions = [0 for _ in range(x.size(0))]
      while not all(len(batch_beam) == 0 for batch_beam in beam):
        successors, forced_completion_successors = self.get_successors(
          x, pointer, beam, beam_size, shift_size)

        new_beam, added_forced_completions = self.reset_beam(
          successors, forced_completion_successors, shift_size > 0)
        for i, c in enumerate(added_forced_completions):
          forced_completions[i] += c

        all_items, batch_idx = self._flatten_items(new_beam)
        all_states = [item.state for item in all_items]
        self.update_stack_rnn(all_items, all_states, word_vecs, batch_idx)

        beam_lengths = [len(batch_beam) for batch_beam in new_beam]
        self.update_beam_and_word_completed(
          beam, word_completed, all_items, beam_lengths, pointer == x.size(1))

        for b in range(len(beam)):
          # Condition to stop beam for current word. This follows original implementation (rnng-bert),
          # but following the paper, the true condition might be
          # `len(word_completed[b]) >= word_beam_size` or
          # `len(word_completed[b]) - forced_completions[b] >= word_beam_size`?
          if len(word_completed[b]) - forced_completions[b] >= beam_size:
            beam[b] = []

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
    return parses, surprisals

  def initial_beam(self, x):
    initial_stack = self.get_initial_stack(x)
    initial_hs = [[(stack_layer[0][b], stack_layer[1][b]) for stack_layer in initial_stack]
                  for b in range(x.size(0))]
    return [[BeamItem.from_initial_stack(h)] for h in initial_hs]

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

    log_probs = self._deflatten_items(log_probs, beam_lengths)
    log_probs = [row.detach().cpu().numpy() for row in log_probs]

    total_scores = self.accumulate_scores(log_probs, beam)  # (batch) -> ((beam, action) -> score)
    successors, forced_completion_successors = self.scores_to_successors(
      x, pointer, beam, total_scores, beam_size, shift_size)

    return successors, forced_completion_successors

  def action_log_probs(self, states, action_mask, next_x = None):
    """
    states: tensor of size (batch, h_dim)
    new_x: tensor of size (batch)
    """
    hiddens = self.stack_top_h(states)  # (total beam size, h_dim)
    hiddens = self.stack_to_hidden(hiddens)  # (total beam size, h_dim)

    action_logit = self.action_mlp(hiddens)
    action_logit[action_mask != 1] = -float('inf')
    log_probs = F.log_softmax(action_logit)  # (total beam size, num_actions)

    if next_x is not None:  # shift is valid for next action
      word_logit = self.vocab_mlp(hiddens)
      shift_idx = self.action_dict.a2i['SHIFT']
      assert next_x.size(0) == hiddens.size(0)
      word_log_probs = self.word_criterion(word_logit, next_x) * -1.0  # (total beam size)
      log_probs[:, shift_idx] += word_log_probs

    return log_probs

  def valid_action_mask(self, items, sent_len):
    mask = items[0].state.stack[0][0][0].new_ones(len(items), self.num_actions, dtype=torch.uint8)
    mask[:, self.action_dict.padding_idx] = 0
    for b, item in enumerate(items):
      state = item.state
      prev_action = item.action
      if state.pointer == sent_len and state.nopen_parens == 0:  # finished
        mask[b,:] = 0
        continue
      if state.nopen_parens == 0:  # only nt
        assert state.pointer == 0 and len(state.stack) == 1
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
          len(state.stack) < 3):
        self.action_dict.mask_reduce(mask, b)

    return mask

  def accumulate_scores(self, action_to_log_prob, beam):
    assert len(action_to_log_prob) == len(beam)  # batch size
    total_scores = []
    for batch, items in enumerate(beam):
      scores = action_to_log_prob[batch]
      assert len(items) == len(scores)
      item_scores = np.expand_dims(np.array([item.score for item in items]), 1)
      total_scores.append(scores + item_scores)
    return total_scores

  def stack_top_h(self, states):
    return torch.stack([state.stack[-1][-1][0] for state in states], dim=0)

  def scores_to_successors(self, x, pointer, beam, total_scores, beam_size, shift_size):
    successors = [[] for _ in range(len(total_scores))]
    forced_completion_successors = [[] for _ in range(len(total_scores))]

    for batch, batch_total_scores in enumerate(total_scores):
      for beam_i, scores in enumerate(batch_total_scores):
        successors[batch] += [(beam_i, a_i, s) for a_i, s in enumerate(scores)]
      if pointer == x.size(1):  # reduce would finish.
        reduce_i = self.action_dict.a2i['REDUCE']
        reduce_succs = [(beam_i, reduce_i, scores[reduce_i]) for beam_i, scores in
                        enumerate(batch_total_scores) if
                        beam[batch][beam_i].state.can_finish_by_reduce()]
        forced_completion_successors[batch] += reduce_succs
      else:  # save all shift
        shift_i = self.action_dict.a2i['SHIFT']
        shift_succs = [(beam_i, shift_i, scores[shift_i]) for beam_i, scores in
                       enumerate(batch_total_scores)]
        forced_completion_successors[batch] += shift_succs

    def remove_invalid(batch_to_succs):
      return [[x for x in succ if x[2] != -float('inf')] for succ in batch_to_succs]

    successors = remove_invalid(successors)
    forced_completion_successors = remove_invalid(forced_completion_successors)

    def sort_by_score(batch_to_succs):
      return [sorted(succ, key=lambda x: x[2], reverse=True) for succ in batch_to_succs]

    successors = sort_by_score(successors)
    forced_completion_successors = sort_by_score(forced_completion_successors)
    successors = [succ[:beam_size] for succ in successors]
    forced_completion_successors = [succ[:shift_size] for succ
                                    in forced_completion_successors]

    def make_succ_item(batch_i, succ):
      beam_i, a_i, score = succ
      return SuccessorItem(beam[batch_i][beam_i], a_i, score)

    def to_succ_item_list(succs):
      return [[make_succ_item(batch_i, succ) for succ in batch_succs]
              for (batch_i, batch_succs) in enumerate(succs)]

    successors = to_succ_item_list(successors)
    forced_completion_successors = to_succ_item_list(forced_completion_successors)

    return successors, forced_completion_successors

  def reset_beam(self, successors, forced_completion_successors, do_force):
    forced_completions = [0 for _ in range(len(successors))]
    if do_force:
      for b in range(len(successors)):
        succ_set = set(successors[b])
        for s in forced_completion_successors[b]:
          if s not in succ_set:
            successors[b].append(s)
            forced_completions[b] += 1

    new_beam = [[succ.to_incomplete_beam_item() for succ in batch_succs]
                for batch_succs in successors]

    return new_beam, forced_completions

  def update_stack_rnn(self, items, states, word_vecs, batch_idx):
    """
    items and states are flattened ones.
    batch_idx is mapping from index in flattend list to the original batch.
    This is necessary to get correct mapping from word_vecs, which is not
    flattened and has size of (batch_size, num words).
    """
    assert len(items) == len(states)
    reduce_idx = [i for i, item in enumerate(items) if
                  self.action_dict.is_reduce(item.action)]
    non_reduce_idx = [i for i, item in enumerate(items) if
                      not self.action_dict.is_reduce(item.action)]

    if len(reduce_idx) > 0:
      reduce_states = [states[i] for i in reduce_idx]
      children, ch_lengths, nt, nt_id = self._collect_children_for_reduce_beam(
        reduce_states)
      reduce_context = self._collect_stack_top_h_beam(reduce_states)
      reduce_context = self.stack_to_hidden(reduce_context)
      # state.stack_top_h(reduce_batch)
      new_child, _, _ = self.composition(children, ch_lengths, nt, nt_id, reduce_context)
    else:
      new_child = None

    new_stack_input = [None] * len(items)
    for i, b in enumerate(reduce_idx):
      new_stack_input[b] = new_child[i]
    for b in non_reduce_idx:
      state = states[b]
      a = items[b].action
      if self.action_dict.is_shift(a):
        new_stack_input[b] = word_vecs[batch_idx[b], state.pointer]  # this pointer is the one before doing aciton, so identical to shifting token position.
      else:
        assert not self.action_dict.is_pad(a)
        nt_id = self.action_dict.nt_id(a)
        new_stack_input[b] = self.nt_emb(word_vecs.new_tensor([nt_id], dtype=torch.long)).squeeze()
    new_stack_input = torch.stack(new_stack_input, 0)
    stack_top_context = self._collect_stack_top_context_beam(states)
    new_stack_top = self.stack_rnn(new_stack_input, stack_top_context)

    for b in range(len(states)):
      states[b].update_stack(new_stack_top, new_stack_input, b)
      items[b].do_action(self.action_dict)

  def update_beam_and_word_completed(self, beam, word_completed, items, beam_lengths, last_token=False):
    accum = 0
    for b in range(len(beam)):
      beam[b] = []
      l = beam_lengths[b]
      for item in items[accum:accum+l]:
        if (last_token and self.action_dict.is_reduce(item.action) and
            item.state.nopen_parens == 0) or self.action_dict.is_shift(item.action):
          # Cases where shifting last+1 token should be pruned by action masking.
          word_completed[b].append(item)
        else:
          beam[b].append(item)
      accum += l

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
    assert accum_idx == items.size(0)
    return de_items  # List[Tensor]

  def _collect_children_for_reduce_beam(self, reduce_states):
    # todo: integrate with _collect_children_for_reduce
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

  def _collect_stack_top_h_beam(self, states):
    return torch.stack([state.stack[-1][-1][0] for state in states], 0)

  def _collect_stack_top_context_beam(self, states):
    stack_h_all = []
    for l in range(self.num_layers):
      h = [state.stack[-1][l][0] for state in states]
      c = [state.stack[-1][l][0] for state in states]
      stack_h_all.append((torch.stack(h, 0), torch.stack(c, 0)))
    return stack_h_all

  def _get_marginal_ll(self, beam):
    ll = []
    for b in range(len(beam)):
      scores = [item.score for item in beam[b]]
      ll.append(torch.logsumexp(torch.tensor(scores), 0).item())
    return ll

class TopDownState(object):
  def __init__(self, initial_stack, sent_len):
    self.sent_len = sent_len
    self.batch_size = initial_stack[0][0].size(0)
    self.stack_top_history = [initial_stack]  # Collection of last stack element at each step. Each element is a list of size (n_layers, 2). 2 is for (h, c). Each list elment is a tensor of size (batch_size, h_dim).

    self.stack = [[] for _ in range(self.batch_size)]  # correspond to stack2
    self.stack_trees = [[] for _ in range(self.batch_size)]  # correspond to stack_child (rather than keeping (h, c), we only keep h, came from composition)
    self.pointer = [0] * self.batch_size
    self.actions = [[] for _ in range(self.batch_size)]
    self.nopen_parens = [0] * self.batch_size
    self.ncons_nts = [0] * self.batch_size
    self.nt_index = [[] for _ in range(self.batch_size)]  # stack of nt index (e.g., [1, 2, 5] means 1,2,5-th elements of stack are open nt)
    self.nt_ids = [[] for _ in range(self.batch_size)]

    self.state_length = [None] * self.batch_size  # will be filled with (# total actions + 1) for each sent.

    for b in range(self.batch_size):
      self.stack[b].append([[stack_layer[0][b], stack_layer[1][b]]
                            for stack_layer in initial_stack])
    self.num_layers = len(self.stack[0][0])

  def action_contexts(self):
    """Transform self.stack_top_history to a tensor of size (batch_size, history_len, h_dim).

    The first and second dimentions should be equal to the size of actions matrix, to allow indexing
    with action values.
    """
    hs = [h[-1][0] for h in self.stack_top_history[:-1]]  # last state is a finished state, which is not used for parameter updates.
    return torch.stack(hs, 1)

  def stack_top_h(self, bs=None):
    # bs: target batch index
    if bs is None:
      bs = range(self.batch_size)
    if isinstance(bs, int):
      bs = [bs]
    return torch.stack([self.stack[b][-1][-1][0] for b in bs], 0)

  def stack_top_context(self):
    """Arranging self.stack to obtain context vector for stack lstm update (with a new element).

    Differently from self.stack_top_h, which returns a single tensor of size (batch_size, h_dim) summarizing
    last layer hidden cells, this method summarizes all layer information, used for update in SeqLSTM.

    self.stack: (batch_size, stack_len, layer, 2) -> (h_dim)
    return: (layer, 2) -> (batch_size, h_dim)
    """
    stack_h_all = []
    for l in range(self.num_layers):
      h = [self.stack[b][-1][l][0] for b in range(self.batch_size)]
      c = [self.stack[b][-1][l][1] for b in range(self.batch_size)]
      stack_h_all.append((torch.stack(h, 0), torch.stack(c, 0)))
    return stack_h_all

  def update_stack(self, new_stack_top, new_tree_elem):
    # new_stack_top: (layer, 2) -> Tensor(batch_size, h_dim)
    # new_tree_elem: Tensor(batch_size, w_dim)
    self.stack_top_history.append(new_stack_top)
    for b in range(self.batch_size):
      if not self.finished(b):
        self.stack[b].append([[layer[0][b], layer[1][b]] for layer in new_stack_top])
        self.stack_trees[b].append(new_tree_elem[b])

  def do_action(self, action, action_dict):
    for b, a in enumerate(action):
      a = a.item()
      self.actions[b].append(a)
      if action_dict.is_shift(a):
        self.pointer[b] += 1
        self.ncons_nts[b] = 0
      elif action_dict.is_nt(a):
        nt_id = action_dict.nt_id(a)
        self.nopen_parens[b] += 1
        self.ncons_nts[b] += 1
        self.nt_index[b].append(len(self.stack[b]) - 1)
        self.nt_ids[b].append(nt_id)
      elif action_dict.is_reduce(a):
        self.nopen_parens[b] -= 1
        self.ncons_nts[b] = 0
        if self.nopen_parens[b] == 0 and self.pointer[b] == self.sent_len:  # finished
          self.state_length[b] = len(self.stack_top_history)
          assert self.state_length[b] == len(self.actions[b]) + 1

  def reduce_stack(self, b):
    open_idx = self.nt_index[b].pop()
    nt_id = self.nt_ids[b].pop()
    self.stack[b] = self.stack[b][:open_idx]
    reduce_trees = self.stack_trees[b][open_idx-1:]
    self.stack_trees[b] = self.stack_trees[b][:open_idx-1]
    assert len(reduce_trees) > 1
    return reduce_trees[0], reduce_trees[1:], nt_id

  def all_finished(self):
    return all([self.finished(b) for b in range(self.batch_size)])

  def finished(self, b):
    return self.state_length[b] is not None

class State:
  def __init__(self,
               pointer = 0,
               stack = [],
               stack_trees = [],
               nopen_parens = 0,
               ncons_nts = 0,
               nt_index = [],
               nt_ids = []):
    self.pointer = pointer
    self.stack = stack
    self.stack_trees = stack_trees
    self.nopen_parens = nopen_parens
    self.ncons_nts = ncons_nts
    self.nt_index = nt_index
    self.nt_ids = nt_ids

  @staticmethod
  def from_initial_stack(initial_stack_elem):
    return State(stack=[initial_stack_elem])

  def can_finish_by_reduce(self):
    return self.nopen_parens == 1

  def copy(self):
    return State(self.pointer, self.stack[:], self.stack_trees[:],
                 self.nopen_parens, self.ncons_nts, self.nt_index[:], self.nt_ids[:])

  def do_action(self, a, action_dict):
    if action_dict.is_shift(a):
      self.pointer += 1
      self.ncons_nts = 0
    elif action_dict.is_nt(a):
      nt_id = action_dict.nt_id(a)
      self.nopen_parens += 1
      self.ncons_nts += 1
      self.nt_index.append(len(self.stack) - 1)
      self.nt_ids.append(nt_id)
    elif action_dict.is_reduce(a):
      self.nopen_parens -= 1
      self.ncons_nts = 0

  def reduce_stack(self):
    open_idx = self.nt_index.pop()
    nt_id = self.nt_ids.pop()
    self.stack = self.stack[:open_idx]
    reduce_trees = self.stack_trees[open_idx-1:]
    self.stack_trees = self.stack_trees[:open_idx-1]
    return reduce_trees[0], reduce_trees[1:], nt_id

  def update_stack(self, new_stack_top, new_tree_elem, b_i):
    self.stack.append([[layer[0][b_i], layer[1][b_i]] for layer in new_stack_top])
    self.stack_trees.append(new_tree_elem[b_i])

class ActionPath:
  def __init__(self, prev=None, action=0, score=0.0):
    self.prev = prev
    self.action = action
    self.score = score

  def add_action(self, action, score):
    return ActionPath(self, action, score)

  def foreach(self, f):
    f(self)
    if self.prev is not None:
      self.prev.foreach(f)

class BeamItem:
  def __init__(self, state, action_path):
    self.state = state
    self.action_path = action_path
    self.action = self.action_path.action
    self.score = self.action_path.score

  def do_action(self, action_dict):
    self.state.do_action(self.action, action_dict)

  @staticmethod
  def from_initial_stack(initial_stack_elem):
    state = State.from_initial_stack(initial_stack_elem)
    path = ActionPath()  # initial action is 0 (pad)
    return BeamItem(state, path)

  def parse_actions(self):
    actions = []
    def add_action(path):
      actions.append(path.action)
    self.action_path.foreach(add_action)
    assert actions[-1] == 0  # last (= initial after revsed) is pad (dummy)
    return list(reversed(actions[:-1]))

class SuccessorItem:
  def __init__(self, prev_beam_item, action, score):
    self.prev_beam_item = prev_beam_item
    self.action = action
    self.score = score

  def __eq__(self, other):
    return (self.prev_beam_item is other.prev_beam_item and
            self.action == other.action and
            self.score == other.score)

  def __hash__(self):
    return hash((id(self.prev_beam_item), self.action, self.score))

  def to_incomplete_beam_item(self):
    """
    This BeamItem is incomplete, as its state is just copy of the previous state.
    Proper action and stack update should be performed.
    """
    state = self.prev_beam_item.state.copy()
    path = self.prev_beam_item.action_path.add_action(self.action, self.score)
    return BeamItem(state, path)
