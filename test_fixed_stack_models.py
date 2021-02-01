import unittest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose, assert_raises
import torch

from fixed_stack_models import *
from fixed_stack_in_order_models import *
from subword_fixed_stack_in_order_models import *
from action_dict import TopDownActionDict, InOrderActionDict

class TestFixedStackModels(unittest.TestCase):

    def test_top_down_rnng_transition(self):
        model = self._get_simple_top_down_model()
        self._test_transition_batch_2(model)

    def _test_transition_batch_2(self, model):
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        stack = model.build_stack(x)

        trees = ["(S  (NP (NP 2 3 ) ) (NP 4 ) )",
                 "(NP (NP 1   ) 2 5 )"]

        actions = self._trees_to_actions(trees)
        a_dict = model.action_dict
        actions = a_dict.mk_action_tensor(actions)
        self.assertEqual(actions.cpu().numpy().tolist(),
                         [[3, 4, 4, 1, 1, 2, 2, 4, 1, 2, 2],
                          [4, 4, 1, 2, 1, 1, 2, 0, 0, 0, 0]])

        word_vecs = model.emb(x)
        hs = word_vecs.new_zeros(actions.size(1), word_vecs.size(0), model.hidden_size)
        hidden_head = stack.hidden_head()[:, :, -1]
        self.assertEqual(hidden_head.size(), (2, model.hidden_size))
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

    def test_rnng_cell_with_beam_dim(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)
        stack = beam.stack
        self.assertEqual(stack.cells.size()[:2], (2, 2))
        batched_actions = [[[3, 4, 1, 1, 2],
                            [4, 4, 1, 1, 2]],
                           [[3, 1, 1, 1, 2],
                            [3, 4, 4, 1, 2]]]
        batched_actions = torch.tensor(batched_actions)
        for i in range(batched_actions.size(2)):
            a = batched_actions[..., i]
            model.rnng(word_vecs, a, stack)
        self.assertTensorAlmostEqual(stack.top_position, torch.tensor([[2, 2], [1, 3]]))
        self.assertTensorAlmostEqual(stack.nt_index_pos, torch.tensor([[0, 0], [-1, 1]]))

    def test_beam_items_do_action(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)
        beam.beam_widths[:] = 2
        batched_actions = [[[3, 4, 1, 1, 2],
                            [4, 4, 1, 1, 2]],
                           [[3, 1, 1, 1, 2],
                            [3, 4, 4, 4, 4]]]
        batched_actions = torch.tensor(batched_actions)
        for i in range(batched_actions.size(2)):
            a = batched_actions[..., i]
            model.rnng(word_vecs, a, beam.stack)
            beam.do_action(a, model.action_dict)
        self.assertTensorAlmostEqual(beam.actions_pos, torch.tensor([[5,5],[5,5]]))
        self.assertTensorAlmostEqual(beam.actions[..., :6],
                                     torch.cat([torch.tensor([-1]).expand(2,2,1), batched_actions], 2))
        self.assertTensorAlmostEqual(beam.ncons_nts, torch.tensor([[0, 0], [0, 5]]))
        self.assertTensorAlmostEqual(beam.nopen_parens, torch.tensor([[1, 1], [0, 5]]))

    def test_beam_items_reconstruct(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 0, 0, 0, 0]))

        orig_hiddens = beam.stack.hiddens.clone()

        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        self.assertTensorAlmostEqual(new_beam_idx[0], torch.tensor([0, 0, 0, 1, 1]))
        self.assertTensorAlmostEqual(new_beam_idx[1], torch.tensor([0, 1, 2, 0, 1]))
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 2]))
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 0], orig_hiddens[0, 0])
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 0], beam.stack.hiddens[0, 1])
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 0], beam.stack.hiddens[0, 2])
        self.assertTensorNotEqual(beam.stack.hiddens[1, 0], beam.stack.hiddens[1, 2])

        orig_hiddens = beam.stack.hiddens.clone()
        actions = new_beam_idx[0].new_full((2, 3), 0)
        actions[new_beam_idx] = torch.tensor([3, 4, 4, 3, 4])  # [[S, NP, NP], [S, NP]]
        model.rnng(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 2]))
        self.assertTensorNotEqual(beam.stack.hiddens[0, 0], orig_hiddens[0, 1])
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 1], beam.stack.hiddens[0, 2])

        # second reconstruction
        reconstruct_idx = (torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([2, 2, 0, 1, 1, 1]))
        orig_hiddens = beam.stack.hiddens.clone()
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        self.assertTensorAlmostEqual(new_beam_idx[0], torch.tensor([0, 0, 0, 1, 1, 1]))
        self.assertTensorAlmostEqual(new_beam_idx[1], torch.tensor([0, 1, 2, 0, 1, 2]))
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 0], orig_hiddens[0, 2])
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 1], orig_hiddens[0, 2])
        self.assertTensorAlmostEqual(beam.stack.hiddens[0, 2], orig_hiddens[0, 0])
        self.assertTensorAlmostEqual(beam.stack.hiddens[1, 0], beam.stack.hiddens[1, 1])
        self.assertTensorAlmostEqual(beam.stack.hiddens[1, 1], beam.stack.hiddens[1, 1])
        self.assertTensorAlmostEqual(beam.stack.hiddens[1, 2], beam.stack.hiddens[1, 1])

    def test_beam_move_items(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 0, 0]))
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        actions = torch.tensor([[3, 4, 1], [4, 1, 3]])  # (S, NP, shift); (NP, shift, S)
        model.rnng(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        empty_move_idx = (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        word_completed_beam.move_items_from(beam, empty_move_idx, torch.tensor([]))
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([0, 0]))

        move_idx = (torch.tensor([0, 0, 1]), torch.tensor([1, 2, 1]))
        move_target_idx = word_completed_beam.move_items_from(beam, move_idx, torch.tensor([-0.5, -0.1, -0.2]))
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([0, 0, 1]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([0, 1, 0]))
        self.assertBeamEqual(beam, word_completed_beam, move_idx, move_target_idx)
        self.assertTensorAlmostEqual(
            word_completed_beam.gen_ll[move_target_idx], torch.tensor([-0.5, -0.1, -0.2]))
        self.assertTensorNotEqual(
            word_completed_beam.gen_ll[move_target_idx], beam.gen_ll[move_idx])

        word_completed_beam.beam_size = 3  # reduce for test purpose.
        # Behavior when beam_widths exceed maximum.
        # Current beam width for word_completed_beam is [2, 1]; beam_size = 3
        # It will discard the last two elements if we try to add additional two elements to 0th batch.
        # Try to add (0, 0), (0, 2), (1, 0), (1, 2)
        move_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2, 0, 2]))
        move_target_idx = word_completed_beam.move_items_from(beam, move_idx)
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([0, 1, 1]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([2, 1, 2]))
        reduced_move_idx = (torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2]))  # remove (0, 2) from move_idx
        self.assertBeamEqual(beam, word_completed_beam, reduced_move_idx, move_target_idx, check_scores = True)

        # Furhter moving elements has no effects.
        move_idx = (torch.tensor([0, 1]), torch.tensor([2, 2]))
        move_target_idx = word_completed_beam.move_items_from(beam, move_idx)
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([3, 3]))
        self.assertTensorAlmostEqual(move_target_idx[0], torch.tensor([]))
        self.assertTensorAlmostEqual(move_target_idx[1], torch.tensor([]))

    def test_beam_shrink(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 3, 1)  # beam_size = 3
        reconstruct_idx = (torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0]))
        new_beam_idx, _ = beam.reconstruct(reconstruct_idx)
        actions = torch.tensor([[3, 4, 1], [4, 0, 0]])  # (S, NP, shift); (NP, shift, S)
        model.rnng(word_vecs, actions, beam.stack)
        beam.do_action(actions, model.action_dict)

        # First, make word_completed_beam a copy of beam.
        # After sorting, the top elements are [[1, 0], [0]]. (if shrinked size is 2)
        beam.gen_ll[reconstruct_idx] = torch.tensor([-4, -3, -10, -6]).float()
        word_completed_beam.move_items_from(beam, reconstruct_idx)

        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([3, 1]))
        word_completed_beam.shrink(2)
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([2, 1]))
        self.assertBeamEqual(beam, word_completed_beam,
                             (torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])),
                             (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])),
                             check_scores=True)

        # Further shrinking has no effects.
        word_completed_beam.shrink(2)
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([2, 1]))
        self.assertBeamEqual(beam, word_completed_beam,
                             (torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])),
                             (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])),
                             check_scores=True)

    def test_invalid_action_mask(self):
        model = self._get_simple_top_down_model(num_nts=2)
        model.max_open_nts = 5
        model.max_cons_nts = 3
        x = torch.tensor([[2, 3], [1, 2]])
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)  # beam_size = 2
        sent_len = torch.tensor([2, 2])
        subword_end_mask = x != 0

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(  # (2, 2, 5); only nt is allowed.
            [[[1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1]],  # beam idx 1 does not exist.
             [[1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1]]]))

        reconstruct_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
        beam.reconstruct(reconstruct_idx)

        def do_action(actions):
            model.rnng(word_vecs, actions, beam.stack)
            beam.do_action(actions, model.action_dict)

        do_action(torch.tensor([[3, 3], [3, 3]]))  # (S, S); (S, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]],
             [[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]]]))

        do_action(torch.tensor([[3, 3], [3, 1]]))  # (S, S); (S, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]],
             [[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]]]))  # still reduce is prohibited (because this is not final token)

        do_action(torch.tensor([[3, 1], [3, 1]]))  # (S, shift); (S, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 1, 1],  # max_cons_nt = 3
              [1, 0, 0, 0, 0]],
             [[1, 0, 1, 1, 1],
              [1, 1, 0, 1, 1]]]))  # reduce is allowed; no shift word

        do_action(torch.tensor([[1, 2], [1, 2]]))  # (shift, r); (shift, r)  # (1, 1) finished
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 0, 0, 0],
              [1, 0, 1, 0, 0]],
             [[1, 0, 0, 0, 0],
              [1, 1, 1, 1, 1]]]))

        do_action(torch.tensor([[3, 1], [1, 0]]))  # (S, shift); (shift, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 0, 0],
              [1, 1, 0, 1, 1]],
             [[1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1]]]))

        do_action(torch.tensor([[3, 2], [2, 0]]))  # (S, r); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 1, 1],  # max_open_nts = 5
              [1, 1, 1, 1, 1]],
             [[1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1]]]))

        do_action(torch.tensor([[1, 0], [2, 0]]))  # (shift, -); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 1],  # max_open_nts = 5
              [1, 1, 1, 1, 1]],
             [[1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1]]]))

        do_action(torch.tensor([[2, 0], [2, 0]]))  # (r, -); (r, -)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 1],  # max_open_nts = 5
              [1, 1, 1, 1, 1]],
             [[1, 1, 1, 1, 1],  # finished
              [1, 1, 1, 1, 1]]]))

    def test_scores_to_successors(self):
        model = self._get_simple_top_down_model(num_nts=2)
        model.max_open_nts = 5
        model.max_cons_nts = 3
        x = torch.tensor([[2, 3], [1, 2]])
        subword_end_mask = x != 0
        word_lengths = torch.tensor([2, 2])
        word_vecs = model.emb(x)
        beam_size, shift_size = 2, 1
        beam, word_completed_beam = model.build_beam_items(x, beam_size, shift_size)
        sent_len = torch.tensor([2, 2])

        def reconstruct_and_do_action(successors, word_completed_successors):
            if word_completed_successors[0][0].size(0) > 0:
                comp_idxs = tuple(word_completed_successors[0][:2])
                moved_idxs = word_completed_beam.move_items_from(
                    beam, comp_idxs, new_gen_ll=word_completed_successors[2])
            new_beam_idxs, _ = beam.reconstruct(successors[0][:2])
            beam.gen_ll[new_beam_idxs] = successors[2]
            actions = successors[1].new_full((x.size(0), beam_size), model.action_dict.padding_idx)
            actions[new_beam_idxs] = successors[1]
            model.rnng(word_vecs, actions, beam.stack)
            beam.do_action(actions, model.action_dict)

        inf = -float('inf')
        scores = torch.tensor(
            [[[inf, inf, inf, -0.3, -0.5],
              [inf, inf, inf, inf, inf]],
             [[inf, inf, inf, -0.6, -0.2],
              [inf, inf, inf, inf, inf]]])

        succs, wc_succs, comps = model.scores_to_successors(x, word_lengths, 0, beam, scores, beam_size, shift_size)
        self.assertTensorAlmostEqual(succs[0][0], torch.tensor([0, 0, 1, 1]))
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 0, 0, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4, 4, 3]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.3, -0.5, -0.2, -0.6]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([]))
        self.assertTensorAlmostEqual(comps, torch.tensor([0, 0]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 2]))

        # This is actually the test for reconstruct_and_do_action defined above
        # (a part of model.beam_step).
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]],
             [[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]]]))

        scores = torch.tensor(
            [[[inf, -0.4, inf, -0.6, -0.8],  # this shift is moved without fast-track.
              [inf, -1.1, inf, -0.7, -0.9]],  # no fast-track because shift_size = 1 and it is already consumed.
             [[inf, -0.9, inf, -0.8, -0.6],  # this shift will be fast-tracked.
              [inf, -1.4, inf, -0.7, -0.4]]])  # this shift will not be saved.

        succs, wc_succs, comps = model.scores_to_successors(x, word_lengths, 0, beam, scores, beam_size, shift_size)
        self.assertTensorAlmostEqual(succs[0][0], torch.tensor([0, 1, 1]))  # shift (-0.1) is moved to wc_succs
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 1, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4, 4]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.6, -0.4, -0.6]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([-0.4, -0.9]))
        self.assertTensorAlmostEqual(comps, torch.tensor([0, 1]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([1, 2]))

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1]],
             [[1, 0, 1, 0, 0],
              [1, 0, 1, 0, 0]]]))
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([1, 1]))

        scores = torch.tensor(
            [[[inf, -1.2, inf, -0.8, -1.0],  # this shift is moved without fast-track.
              [inf, inf, inf, inf, inf]],  # no fast-track because shift_size = 1 and it is already consumed.
             [[inf, inf, inf, inf, inf],  # to test finished batch.
              [inf, inf, inf, inf, inf]]])

        succs, wc_succs, comps = model.scores_to_successors(x, word_lengths, 0, beam, scores, beam_size, shift_size)
        self.assertTensorAlmostEqual(succs[0][0], torch.tensor([0, 0]))  # shift (-0.1) is moved to wc_succs
        self.assertTensorAlmostEqual(succs[0][1], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(succs[1], torch.tensor([3, 4]))
        self.assertTensorAlmostEqual(succs[2], torch.tensor([-0.8, -1.0]))

        self.assertTensorAlmostEqual(wc_succs[0][0], torch.tensor([0]))
        self.assertTensorAlmostEqual(wc_succs[0][1], torch.tensor([0]))
        self.assertTensorAlmostEqual(wc_succs[1], torch.tensor([1]))
        self.assertTensorAlmostEqual(wc_succs[2], torch.tensor([-1.2]))
        self.assertTensorAlmostEqual(comps, torch.tensor([1, 0]))
        reconstruct_and_do_action(succs, wc_succs)
        self.assertTensorAlmostEqual(word_completed_beam.beam_widths, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 0]))

        model.finalize_word_completed_beam(x, subword_end_mask, word_lengths, word_vecs, 0, beam, word_completed_beam, 2)
        self.assertTensorAlmostEqual(beam.beam_widths, torch.tensor([2, 1]))
        self.assertTensorAlmostEqual(
            beam.gen_ll[(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]))],
            torch.tensor([-0.4, -1.2, -0.9]))

    def test_beam_search(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        subword_end_mask = x != 0
        parses, surprisals = model.word_sync_beam_search(x, subword_end_mask, 8, 5, 1)
        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)

        paths = set([tuple(parse) for parse, score in parses[0]])
        self.assertEqual(len(paths), 5)

        for parse, score in parses[0]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[0])
        self.assertEqual([len(s) for s in surprisals], [3, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def test_beam_search_different_length(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4, 1, 3], [1, 2, 5, 0, 0]])
        subword_end_mask = x != 0
        parses, surprisals = model.word_sync_beam_search(x, subword_end_mask, 8, 5, 1)
        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)
        self.assertEqual(len(parses[1]), 5)

        for parse, score in parses[1]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[1])
        self.assertEqual([len(s) for s in surprisals], [5, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[1]))

        for parse, score in parses[0]:
            self.assertEqual(len([a for a in parse if a == 1]), 5)  # 1 = shift
        for parse, score in parses[1]:
            self.assertEqual(len([a for a in parse if a == 1]), 3)

    def test_in_order_invalid_action_mask(self):
        model = self._get_simple_in_order_model(num_nts=2)
        model.max_open_nts = 2
        model.max_cons_nts = 2
        x = torch.tensor([[2, 3], [1, 2]])
        subword_end_mask = x != 0
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)  # beam_size = 2
        sent_len = torch.tensor([2, 2])

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(  # (2, 2, 5); only shift is allowed.
            [[[1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1]],  # beam idx 1 does not exist.
             [[1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1]]]))

        reconstruct_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
        beam.reconstruct(reconstruct_idx)

        def do_action(actions):
            model.rnng(word_vecs, actions, beam.stack, subword_end_mask)
            beam.do_action(actions, model.action_dict)

        do_action(torch.tensor([[1, 1], [1, 1]]))  # (shift, shift); (shift, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]],
             [[1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]]]))

        do_action(torch.tensor([[4, 4], [4, 4]]))  # (S, S); (S, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 0, 1, 1, 1],
              [1, 0, 0, 1, 1, 1]],
             [[1, 0, 0, 1, 1, 1],
              [1, 0, 0, 1, 1, 1]]]))

        do_action(torch.tensor([[2, 1], [2, 1]]))  # (r, shift); (r, shift)  # reset ncons_nt
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 1, 1, 0, 0],  # (S w)
              [1, 1, 0, 1, 0, 0]],  # (S w w
             [[1, 1, 1, 1, 0, 0],  # (S w)
              [1, 1, 0, 1, 0, 0]]]))  # (S w w

        do_action(torch.tensor([[4, 2], [4, 4]]))  # (S, r); (S, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 1, 1, 1],  # (S (S w)  ncons_nt=2 -> no more reduce
              [1, 1, 1, 0, 0, 0]],  # (S w w)
             [[1, 0, 1, 1, 1, 1],  # (S (S w)  ncons_nt=2 -> no more reduce
              [1, 1, 0, 1, 1, 1]]]))  # (S w (S w

        do_action(torch.tensor([[1, 3], [1, 2]]))  # (shift, finish); (shift, r)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 0, 0],  # (S (S w) w
              [1, 1, 1, 1, 1, 1]],  # (S w w) finish
             [[1, 1, 0, 1, 0, 0],  # (S (S w) w
              [1, 1, 0, 1, 0, 0]]]))  # (S w (S w)

        do_action(torch.tensor([[4, 0], [4, 4]]))  # (S, -); (S, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 1, 1],  # (S (S w) (S w
              [1, 1, 1, 1, 1, 1]],  # (S w w) finish
             [[1, 1, 0, 1, 1, 1],  # (S (S w) (S w
              [1, 1, 0, 1, 1, 1]]]))  # (S w (S (S w) ncons_nt=2 -> still can reduce (since sentence final)

        do_action(torch.tensor([[2, 0], [2, 2]]))  # (r, -); (r, r)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 0, 0],  # (S (S w) (S w)
              [1, 1, 1, 1, 1, 1]],
             [[1, 1, 0, 1, 0, 0],  # (S (S w) (S w)   # max_cons_nts = 3 -> not sent end -> still can open
              [1, 1, 0, 1, 1, 1]]]))  # (S w (S (S w)) ncons_nt=2 -> no more nt

        do_action(torch.tensor([[4, 0], [2, 4]]))  # (S, -); (r, S)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 1, 1],  # (S (S w) (S (S w)
              [1, 1, 1, 1, 1, 1]],
             [[1, 1, 1, 0, 0, 0],  # (S (S w) (S w))   # max_cons_nts = 3 -> no more reduce
              [1, 1, 0, 1, 1, 1]]]))  # (S w (S (S (S w))

    def test_in_order_beam_search(self):
        model = self._get_simple_in_order_model()
        x = torch.tensor([[1, 3, 4], [2, 2, 5]])
        subword_end_mask = x != 1
        parses, surprisals = model.word_sync_beam_search(x, subword_end_mask, 8, 5, 1)
        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)

        paths = set([tuple(parse) for parse, score in parses[0]])
        self.assertEqual(len(paths), 5)

        for parse, score in parses[0]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[0])
        self.assertEqual([len(s) for s in surprisals], [3, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def test_variable_beam_search(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        subword_end_mask = x > 0
        parses, surprisals = model.variable_beam_search(x, subword_end_mask, 50)

        self.assertEqual(len(parses), 2)
        self.assertTrue(len(parses[0]) > 0)

        paths = set([tuple(parse) for parse, score in parses[0]])
        self.assertEqual(len(paths), len(parses[0]))

        for parse, score in parses[0]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[0])
        self.assertEqual([len(s) for s in surprisals], [3, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def test_variable_beam_search_different_length(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4, 1, 3], [1, 2, 5, 0, 0]])
        subword_end_mask = x != 1
        parses, surprisals = model.variable_beam_search(x, subword_end_mask, 50)
        self.assertEqual(len(parses), 2)

        self.assertEqual(len(set([tuple(parse) for parse, score in parses[0]])), len(parses[0]))
        self.assertEqual(len(set([tuple(parse) for parse, score in parses[1]])), len(parses[1]))

        for parse, score in parses[1]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[1])
        self.assertEqual([len(s) for s in surprisals], [5, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[1]))

        for parse, score in parses[0]:
            self.assertEqual(len([a for a in parse if a == 1]), 5)  # 1 = shift
        for parse, score in parses[1]:
            self.assertEqual(len([a for a in parse if a == 1]), 3)

    def test_subword_in_order_invalid_action_mask(self):
        model = self._get_simple_in_order_model(num_nts=2)
        model.max_open_nts = 2
        model.max_cons_nts = 2
        subword_end_mask = torch.tensor([[True, False, True], [False, True, True]])
        x = torch.tensor([[2, 1, 3], [1, 2, 3]])  # 1 is an intermediate word
        word_vecs = model.emb(x)
        beam, word_completed_beam = model.build_beam_items(x, 2, 1)  # beam_size = 2
        sent_len = torch.tensor([3, 3])

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(  # (2, 2, 5); only shift is allowed.
            [[[1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1]],  # beam idx 1 does not exist.
             [[1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1]]]))

        reconstruct_idx = (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0]))
        beam.reconstruct(reconstruct_idx)

        def do_action(actions):
            model.rnng(word_vecs, actions, beam.stack, subword_end_mask)
            beam.do_action(actions, model.action_dict)

        do_action(torch.tensor([[1, 1], [1, 1]]))  # (shift, shift); (shift, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]],
             [[1, 0, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1]]]))

        do_action(torch.tensor([[4, 4], [1, 1]]))  # (S, S); (shift, shift)

        # Test the behavior of new do_nt.
        self.assertTensorAlmostEqual(beam.stack.nt_index_pos[0], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(beam.stack.nt_index[0,:,0], torch.tensor([1, 1]))
        # Test that swap is correctly done.
        self.assertTensorAlmostEqual(beam.stack.trees[0,0,1], word_vecs[0,0])
        self.assertTensorAlmostEqual(beam.stack.trees[0,0,0],
                                     model.rnng.nt_emb(torch.tensor([0]))[0])
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 0, 1, 1, 1],  # (S w_
              [1, 0, 0, 1, 1, 1]],  # (S w_
             [[1, 1, 1, 1, 0, 0],  # w w_
              [1, 1, 1, 1, 0, 0]]]))  # w w_

        do_action(torch.tensor([[2, 1], [4, 4]]))  # (r, shift); (S, S)

        # Test that swap is correctly done.
        self.assertTensorAlmostEqual(beam.stack.nt_index_pos[1], torch.tensor([0, 0]))
        self.assertTensorAlmostEqual(beam.stack.nt_index[1,:,0], torch.tensor([1, 1]))
        self.assertTensorAlmostEqual(beam.stack.trees[1,0,1:3], word_vecs[1,:2])
        self.assertTensorAlmostEqual(beam.stack.trees[1,0,0],
                                     model.rnng.nt_emb(torch.tensor([0]))[0])

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 1, 1, 0, 0],  # (S w_)
              [1, 0, 1, 1, 1, 1]],  # (S w_ w  no_end_word->shift_only
             [[1, 0, 0, 1, 1, 1],  # (S w w_
              [1, 0, 0, 1, 1, 1]]]))  # (S w w_

        do_action(torch.tensor([[4, 1], [1, 2]]))  # (S, shift); (shift, r)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 1, 1, 1],  # (S (S w_)  ncons_nt=2 -> no more reduce
              [1, 1, 0, 1, 0, 0]],  # (S w_ w w_
             [[1, 1, 0, 1, 0, 0],  # (S w w_ w_
              [1, 1, 1, 1, 0, 0]]]))  # (S w w_)

        do_action(torch.tensor([[1, 2], [5, 5]]))  # (shift, r); (NP, NP)

        # Test that swap is correctly done.
        self.assertTensorAlmostEqual(beam.stack.nt_index_pos[1], torch.tensor([1, 0]))
        self.assertEqual(beam.stack.nt_index[1,0,0], 1)
        self.assertEqual(beam.stack.nt_index[1,0,1], 4)
        self.assertEqual(beam.stack.nt_index[1,1,0], 1)
        self.assertTensorAlmostEqual(beam.stack.trees[1,0,4], word_vecs[1,2])
        self.assertTensorAlmostEqual(beam.stack.trees[1,0,3],
                                     model.rnng.nt_emb(torch.tensor([1]))[0])
        self.assertTensorAlmostEqual(beam.stack.trees[1,1,0],
                                     model.rnng.nt_emb(torch.tensor([1]))[0])

        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 0, 1, 1, 1, 1],  # (S (S w_) w
              [1, 1, 1, 0, 0, 0]],  # (S w_ w w_)
             [[1, 1, 0, 1, 1, 1],  # (S w w_ (NP w_
              [1, 0, 1, 1, 1, 1]]]))  # (NP (S w w_)  ncons_nt=2 -> no more reduce

        self.assertEqual(beam.stack.last_subword_begin_idx[0,0], 3)

        do_action(torch.tensor([[1, 3], [2, 1]]))  # (shift, finish); (r, shift)
        mask = model.invalid_action_mask(beam, sent_len, subword_end_mask)
        self.assertTensorAlmostEqual(mask, torch.tensor(
            [[[1, 1, 0, 1, 0, 0],  # (S (S w_) w w_
              [1, 1, 1, 1, 1, 1]],  # -
             [[1, 1, 0, 1, 0, 0],  # (S w w_ (NP w_)
              [1, 1, 0, 1, 0, 0]]]))  # (NP (S w w_) w_

        self.assertEqual(beam.stack.last_subword_begin_idx[0,0], 3)

        do_action(torch.tensor([[5, 0], [4, 2]]))  # (NT, -); (S, r)

        # Test that swap is correctly done.
        self.assertTensorAlmostEqual(beam.stack.nt_index_pos[0], torch.tensor([1, -1]))
        self.assertEqual(beam.stack.nt_index[0,0,0], 1)  # *(S* (S w_) (NP w w_
        self.assertEqual(beam.stack.nt_index[0,0,1], 3)  # (S (S w_) *(NP* w w_
        self.assertTensorAlmostEqual(beam.stack.trees[0,0,3:5], word_vecs[0,1:])
        self.assertTensorAlmostEqual(beam.stack.trees[0,0,2],
                                     model.rnng.nt_emb(torch.tensor([1]))[0])

    

    def _get_simple_top_down_model(self, vocab=6, w_dim=4, h_dim=6, num_layers=2, num_nts=2):
        nts = ['S', 'NP', 'VP', 'X3', 'X4', 'X5', 'X6'][:num_nts]
        a_dict = TopDownActionDict(nts)
        return FixedStackRNNG(a_dict, vocab=vocab, w_dim=w_dim, h_dim=h_dim, num_layers=num_layers)

    def _get_simple_in_order_model(self, vocab=6, w_dim=4, h_dim=6, num_layers=2, num_nts=2):
        nts = ['S', 'NP', 'VP', 'X3', 'X4', 'X5', 'X6'][:num_nts]
        a_dict = InOrderActionDict(nts)
        return FixedStackInOrderRNNG(a_dict, vocab=vocab, w_dim=w_dim, h_dim=h_dim, num_layers=num_layers)

    def _trees_to_actions(self, trees):
        def conv(a):
            if a[0] ==  '(':
                return 'NT({})'.format(a[1:])
            elif a == ')':
                return 'REDUCE'
            else:
                return 'SHIFT'
        return [[conv(x) for x in tree.split()] for tree in trees]

    def _random_beam_items(self, model, x, stack_size = 5, max_actions = 10,
                           beam_size = 3, beam_is_empty = False):
        stack = FixedStack(model.rnng.get_initial_hidden(x), stack_size, model.input_size, beam_size)
        beam = BeamItems(stack, max_actions, beam_is_empty)

    def assertBeamEqual(self, beam1, beam2, idx1, idx2, check_scores = False):
        attrs = ['actions', 'actions_pos', 'ncons_nts', 'nopen_parens']
        if check_scores:
            attrs += ['gen_ll']
        stack_attrs = ['pointer', 'top_position', 'hiddens', 'cells', 'trees',
                       'nt_index', 'nt_ids', 'nt_index_pos']
        for a in attrs:
            self.assertTensorAlmostEqual(getattr(beam1, a)[idx1], getattr(beam2, a)[idx2])
        for a in stack_attrs:
            self.assertTensorAlmostEqual(getattr(beam1.stack, a)[idx1], getattr(beam2.stack, a)[idx2])

    def assertTensorAlmostEqual(self, x, y):
        self.assertIsNone(assert_almost_equal(x.cpu().detach().numpy(), y.cpu().detach().numpy()))

    def assertTensorNotEqual(self, x, y):
        self.assertIsNone(assert_raises(AssertionError, assert_almost_equal,
                                        x.cpu().detach().numpy(), y.cpu().detach().numpy()))
