import unittest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import torch

from fixed_stack_models import *
from in_order_models import *
from action_dict import TopDownActionDict, InOrderActionDict

class TestFixedStackModels(unittest.TestCase):

    def test_top_down_rnng_transition(self):
        model = self._get_simple_top_down_model()
        self._test_transition_batch_2(model)

    def _test_transition_batch_2(self, model):
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        stack = model.build_stack(x)

        trees = ["(S  (NP (NP 2 3 ) ) (VP 4 ) )",
                 "(NP (NP 1   ) 2 5 )"]

        actions = self._trees_to_actions(trees)
        a_dict = model.action_dict
        actions = a_dict.mk_action_tensor(actions)
        self.assertEqual(actions.cpu().numpy().tolist(),
                         [[3, 4, 4, 1, 1, 2, 2, 5, 1, 2, 2],
                          [4, 4, 1, 2, 1, 1, 2, 0, 0, 0, 0]])

        word_vecs = model.emb(x)
        hs = word_vecs.new_zeros(actions.size(1), word_vecs.size(0), model.input_size)
        hidden_head = model.rnng.output(stack.hidden_head()[:, :, -1])
        self.assertEqual(hidden_head.size(), (2, model.input_size))
        hs[0] = hidden_head
        hiddens = []
        hiddens.append(stack.hiddens.clone())

        hidden = model.rnng(word_vecs, actions[:, 0], stack)
        hs[1] = hidden
        hiddens.append(stack.hiddens.clone())  # (batch_size, max_stack, hidden_size, num_layers)
        # stack should be updated
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(stack.nt_index[:,:1], torch.tensor([[1], [1]]))

        nt = [a_dict.nt_id(i) for i in range(5)]
        self.assertTensorAlmostEqual(stack.nt_ids[:,:1], torch.tensor([[nt[3]], [nt[4]]]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([0, 0]))

        hidden = model.rnng(word_vecs, actions[:, 1], stack)
        hs[2] = hidden
        hiddens.append(stack.hiddens.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([2, 2]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(stack.nt_index[:,:2], torch.tensor([[1, 2], [1, 2]]))
        self.assertTensorAlmostEqual(stack.nt_ids[:,:2], torch.tensor([[nt[3], nt[4]], [nt[4], nt[4]]]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([1, 1]))

        hidden = model.rnng(word_vecs, actions[:, 2], stack)
        hs[3] = hidden
        hiddens.append(stack.hiddens.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(stack.nt_index[:,:3], torch.tensor([[1, 2, 3], [1, 2, 0]]))
        self.assertTensorAlmostEqual(stack.nt_ids[:,:3], torch.tensor([[nt[3], nt[4], nt[4]], [nt[4], nt[4], 0]]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([2, 1]))

        hidden = model.rnng(word_vecs, actions[:, 3], stack)  # (SHIFT, REDUCE)
        hs[4] = hidden
        hiddens.append(stack.hiddens.clone())
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([4, 2]))
        self.assertTensorAlmostEqual(stack.pointer, torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(stack.nt_index[:,:3], torch.tensor([[1, 2, 3], [1, 2, 0]]))
        self.assertTensorAlmostEqual(stack.nt_ids[:,:3], torch.tensor([[nt[3], nt[4], nt[4]], [nt[4], nt[4], 0]]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([2, 0]))

        self.assertTrue(hiddens[3][1, 2].sum() != hiddens[4][1, 2].sum())

        hs[5] = model.rnng(word_vecs, actions[:, 4], stack)  # (SHIFT, SHIFT)
        hs[6] = model.rnng(word_vecs, actions[:, 5], stack)  # (REDUCE, SHIFT)
        hs[7] = model.rnng(word_vecs, actions[:, 6], stack)  # (REDUCE, REDUCE)

        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([0, -1]))
        self.assertTensorAlmostEqual(stack.nt_ids[:,:1], torch.tensor([[nt[3]], [nt[4]]]))

        hs[8] = model.rnng(word_vecs, actions[:, 7], stack)  # (NT(VP), <pad>)
        hs[9] = model.rnng(word_vecs, actions[:, 8], stack)  # (SHIFT, <pad>)
        hs[10] = model.rnng(word_vecs, actions[:, 9], stack)  # (REDUCE, <pad>)
        model.rnng(word_vecs, actions[:, 10], stack)  # (REDUCE, <pad>)

        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([1, 1]))

        # a_loss, _ = model.action_loss(actions, a_dict, action_contexts)
        # w_loss, _ = model.word_loss(x, actions, a_dict, action_contexts)

        # self.assertEqual(a_loss.size(), (18,))

    def _get_simple_top_down_model(self, vocab=6, w_dim=4, h_dim=6, num_layers=2):
        a_dict = TopDownActionDict(['S', 'NP', 'VP', 'PP'])
        return FixedStackRNNG(a_dict, vocab=vocab, input_size=w_dim, hidden_size=h_dim, num_layers=num_layers)

    def _trees_to_actions(self, trees):
        def conv(a):
            if a[0] ==  '(':
                return 'NT({})'.format(a[1:])
            elif a == ')':
                return 'REDUCE'
            else:
                return 'SHIFT'
        return [[conv(x) for x in tree.split()] for tree in trees]

    def assertTensorAlmostEqual(self, x, y):
        self.assertIsNone(assert_almost_equal(x.cpu().detach().numpy(), y.cpu().detach().numpy()))
