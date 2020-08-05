
import torch

class TopDownActionDict:
    def __init__(self, nonterminals):
        assert isinstance(nonterminals, list)
        self.nonterminals = nonterminals
        self.i2a = ['<pad>', 'SHIFT', 'REDUCE'] + ['NT({})'.format(nt) for nt in nonterminals]
        self.a2i = dict([(a, i) for i, a in enumerate(self.i2a)])
        self.padding_idx = 0

    def to_id(self, actions):
        return [self.a2i[a] for a in actions]

    def num_actions(self):
        return len(self.i2a)

    def num_nts(self):
        return len(self.i2a) - 3

    def mask_shift(self, mask, batch_i):
        mask[batch_i][1] = 0

    def mask_reduce(self, mask, batch_i):
        mask[batch_i][2] = 0

    def mask_nt(self, mask, batch_i):
        mask[batch_i][3:] = 0

    def is_pad(self, a):
        return a == 0

    def is_shift(self, a):
        return a == 1

    def is_reduce(self, a):
        return a == 2

    def is_nt(self, a):
        return a > 2

    def nt_id(self, a):
        return a - 3

    def mk_action_tensor(self, action_strs, device='cpu'):
        action_ids = [[self.a2i[a] for a in action_str] for action_str in action_strs]
        max_len = max([len(ids) for ids in action_ids])
        for i in range(len(action_ids)):
            action_ids[i] += [self.padding_idx] * (max_len-len(action_ids[i]))

        return torch.tensor(action_ids, device=device)

    def build_tree_str(self, actions, tokens, tags):
        ret = ''
        tok_i = 0
        for a in actions:
            if self.is_nt(a):
                ret += ' ( {} '.format(self.nonterminals[self.nt_id(a)])
            elif self.is_shift(a):
                ret += ' ( {} {} ) '.format(tags[tok_i], tokens[tok_i])
                tok_i += 1
            elif self.is_reduce(a):
                ret += ' ) '

        return ret.replace(' ( ', '(').replace(' ) ', ')').replace(')(', ') (')
