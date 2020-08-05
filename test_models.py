import unittest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import torch

from models import *
from action_dict import TopDownActionDict

class TestModels(unittest.TestCase):

    def test_lstm_composition(self):
        w_dim = 10
        comp = LSTMComposition(w_dim, 0.1)

        batch_size = 3
        children, ch_lengths = self._random_children(batch_size, w_dim)
        nt_id, stack_state = None, None
        nt = torch.rand(batch_size, w_dim)

        result, _, _ = comp(children, ch_lengths, nt, nt_id, stack_state)
        # print(result)

        self.assertTrue(result.size(), (batch_size, w_dim))
        self.assertTrue(result.sum() != 0)

    def test_attention_composition(self):
        w_dim, h_dim = 2, 3
        num_labels = 8
        comp = AttentionComposition(w_dim, h_dim, num_labels)

        batch_size = 3
        children, ch_lengths = self._random_children(batch_size, w_dim)
        nt_id = torch.tensor([0, 5, 2])
        stack_state = torch.rand(batch_size, h_dim)
        nt = None

        result, attn, gate = comp(children, ch_lengths, nt, nt_id, stack_state)
        # print(result)

        for b, l in enumerate(ch_lengths):
            self.assertTrue(all(attn[b,l:] == 0.0))
            self.assertAlmostEqual(attn[b,:l].sum().item(), 1.0, delta=1e-4)

        self.assertTrue(result.size(), (batch_size, w_dim))
        self.assertTrue(result.sum() != 0)

    def _random_children(self, batch_size, w_dim, lengths = [2, 4, 1]):
        ch_lengths = torch.tensor(lengths)
        children = torch.zeros(batch_size, max(ch_lengths), w_dim)
        for b, l in enumerate(ch_lengths):
            children[b, :l] = torch.rand(l, w_dim)
        return children, ch_lengths

    # def test_top_down_rnng_initialize(self):
    #     model = TopDownRNNG()

    def test_top_down_state_init(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        initial_state = model.get_initial_stack(x)
        state = TopDownState(initial_state, x.size(1))

        self.assertEqual(state.batch_size, 2)
        self.assertEqual(state.num_layers, 2)
        self.assertFalse(state.all_finished())
        initial_second_layer = state.stack[0][0][1]
        self.assertEqual(len(initial_second_layer), 2)
        self.assertEqual(initial_second_layer[0].size(0), 6)
        self.assertTrue(all(initial_second_layer[0] == initial_state[1][0][0]))

        top_context = state.stack_top_context()
        self.assertEqual(len(top_context), 2)
        self.assertEqual(len(top_context[0]), 2)
        self.assertEqual(top_context[0][0].size(), (2, 6))

    def test_top_down_rnng_transition(self):
        model = self._get_simple_top_down_model()
        self._test_transition_batch_2(model)

    def test_top_down_rnng_transition_tied_weight(self):
        model = self._get_simple_top_down_model(w_dim=10, h_dim=10)
        self._test_transition_batch_2(model)

    def _test_transition_batch_2(self, model):
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        initial_state = model.get_initial_stack(x)
        state = TopDownState(initial_state, x.size(1))

        trees = ["(S  (NP (NP 2 3 ) ) (VP 4 ) )",
                 "(NP (NP 1   ) 2 5 )"]

        actions = self._trees_to_actions(trees)
        a_dict = model.action_dict
        actions = a_dict.mk_action_tensor(actions)
        self.assertEqual(actions.cpu().numpy().tolist(),
                         [[3, 4, 4, 1, 1, 2, 2, 5, 1, 2, 2],
                          [4, 4, 1, 2, 1, 1, 2, 0, 0, 0, 0]])

        word_vecs = model.emb(x)
        model.one_step_for_train(state, word_vecs, actions[:, 0])  # (NT(S), NT(NP))
        self.assertTrue(len(state.stack[0]) == len(state.stack[1]) == 2)
        self.assertTrue(len(state.stack_trees[0]) == len(state.stack_trees[1]) == 1)
        self.assertTensorAlmostEqual(state.stack_trees[0][-1], model.nt_emb(torch.LongTensor([0]))[0])
        self.assertTensorAlmostEqual(state.stack[0][1][0][0], state.stack_top_history[1][0][0][0])
        self.assertEqual(state.nopen_parens, [1, 1])
        self.assertEqual(state.nt_index, [[1], [1]])
        self.assertEqual(state.nt_ids, [[a_dict.nt_id(3)], [a_dict.nt_id(4)]])

        model.one_step_for_train(state, word_vecs, actions[:, 1])  # (NT(NP), NT(NP))
        self.assertTrue(len(state.stack[0]) == len(state.stack[1]) == 3)
        self.assertTrue(len(state.stack_trees[0]) == len(state.stack_trees[1]) == 2)
        self.assertEqual(state.nopen_parens, [2, 2])
        self.assertEqual(state.nt_index, [[1, 2], [1, 2]])
        self.assertEqual(state.nt_ids, [[a_dict.nt_id(3), a_dict.nt_id(4)],
                                        [a_dict.nt_id(4), a_dict.nt_id(4)]])

        model.one_step_for_train(state, word_vecs, actions[:, 2])  # (NT(NP), SHIFT)
        self.assertTrue(len(state.stack[0]) == len(state.stack[1]) == 4)
        self.assertTrue(len(state.stack_trees[0]) == len(state.stack_trees[1]) == 3)
        self.assertEqual(state.nopen_parens, [3, 2])
        self.assertEqual(state.ncons_nts, [3, 0])
        self.assertEqual(state.nt_index, [[1, 2, 3], [1, 2]])
        self.assertEqual(state.nt_ids, [[a_dict.nt_id(3), a_dict.nt_id(4), a_dict.nt_id(4)],
                                        [a_dict.nt_id(4), a_dict.nt_id(4)]])

        model.one_step_for_train(state, word_vecs, actions[:, 3])  # (SHIFT, REDUCE)
        self.assertTrue(len(state.stack[0]) == 5)
        self.assertTrue(len(state.stack[1]) == 3)
        self.assertTrue(len(state.stack_trees[0]) == 4)
        self.assertTrue(len(state.stack_trees[1]) == 2)
        self.assertEqual(state.nopen_parens, [3, 1])
        self.assertEqual(state.ncons_nts, [0, 0])
        self.assertEqual(state.nt_index, [[1, 2, 3], [1]])
        self.assertEqual(state.nt_ids, [[a_dict.nt_id(3), a_dict.nt_id(4), a_dict.nt_id(4)],
                                        [a_dict.nt_id(4)]])

        model.one_step_for_train(state, word_vecs, actions[:, 4])  # (SHIFT, SHIFT)
        model.one_step_for_train(state, word_vecs, actions[:, 5])  # (REDUCE, SHIFT)
        model.one_step_for_train(state, word_vecs, actions[:, 6])  # (REDUCE, REDUCE)
        self.assertTrue(len(state.stack[0]) == 3)
        self.assertTrue(len(state.stack[1]) == 2)
        self.assertTrue(len(state.stack_trees[0]) == 2)
        self.assertTrue(len(state.stack_trees[1]) == 1)
        self.assertEqual(state.nopen_parens, [1, 0])
        self.assertEqual(state.ncons_nts, [0, 0])
        self.assertEqual(state.nt_index, [[1], []])
        self.assertEqual(state.nt_ids, [[a_dict.nt_id(3)], []])
        self.assertTrue(state.finished(1))

        model.one_step_for_train(state, word_vecs, actions[:, 7])  # (NT(VP), <pad>)
        model.one_step_for_train(state, word_vecs, actions[:, 8])  # (SHIFT, <pad>)
        model.one_step_for_train(state, word_vecs, actions[:, 9])  # (REDUCE, <pad>)
        model.one_step_for_train(state, word_vecs, actions[:, 10])  # (REDUCE, <pad>)

        self.assertTrue(state.all_finished())
        action_contexts = state.action_contexts()
        self.assertEqual(action_contexts.size()[:2], actions.size())

        a_loss, _ = model.action_loss(actions, a_dict, action_contexts)
        w_loss, _ = model.word_loss(x, actions, a_dict, action_contexts)

        self.assertEqual(a_loss.size(), (18,))

    def test_action_path(self):
        start = ActionPath()
        end = start.add_action(1, 1.0).add_action(3, 3.0).add_action(2, 5.0)
        item = BeamItem(None, end)
        self.assertEqual(item.parse_actions(), [1, 3, 2])

    def test_successor_item(self):
        state1 = State()
        state2 = state1.copy()
        state2.nt_index.append(3)
        self.assertTrue(state1.nt_index != state2.nt_index)

        item = BeamItem(state2, ActionPath(None, 2, 2.0))
        item2 = BeamItem(state1, ActionPath(None, 2, 2.0))

        succ1 = SuccessorItem(item, 1, 1.0)
        succ2 = SuccessorItem(item, 1, 1.0)
        succ3 = SuccessorItem(item2, 1, 1.0)

        self.assertEqual(succ1, succ2)
        self.assertNotEqual(succ1, succ3)

        succ_set = set([succ1, succ3])
        self.assertTrue(succ2 in succ_set)

        item_dummy = BeamItem(state2, ActionPath(None, 2, 2.0))
        succ1_dummy = SuccessorItem(item_dummy, 1, 1.0)
        self.assertNotEqual(succ1_dummy, succ1)  # Even with the same content (numbers), two succs are different if addresses of BeamItem are different.

    def test_reset_beam(self):
        model = self._get_simple_top_down_model()
        state = State()
        item1 = BeamItem(state, ActionPath(None, 2, 2.0))
        item2 = BeamItem(state, ActionPath(None, 2, 2.0))
        succ1 = SuccessorItem(item1, 1, 1.0)
        succ2 = SuccessorItem(item2, 1, 1.0)
        succ3 = SuccessorItem(item1, 1, 1.0)
        new_beam, forced_completions = model.reset_beam([[succ1]], [[succ2, succ3]], True)
        self.assertEqual([len(b) for b in new_beam], [2])
        self.assertEqual(forced_completions, [1])

    def test_get_successors(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        initial_beam = model.initial_beam(x)
        beam = initial_beam
        successors, forced_completion_successors = model.get_successors(x, 0, beam, 5, 2)
        self.assertEqual(forced_completion_successors, [[], []])  # shift is prohibited for initial action
        self.assertEqual([len(s) for s in successors], [4, 4])
        self.assertTrue(all((successors[0][i].score > successors[0][i+1].score and
                             successors[0][i].score != -float('inf'))
                            for i in range(len(successors[0])-1)))

        def simulate(actions):
            return self._simulate_beam_item(
                initial_beam[0][0], actions, model.action_dict, model.w_dim)

        # shift is valid after an NT
        beam = [[simulate(['NT(S)'])],
                [simulate(['NT(NP)', 'NT(NP)']),
                 simulate(['NT(NP)', 'NT(VP)'])]]
        successors, forced_completion_successors = model.get_successors(x, 0, beam, 8, 2)
        self.assertEqual([len(s) for s in forced_completion_successors], [1, 2])
        self.assertEqual([len(s) for s in successors], [5, 8])
        self.assertTrue(all((successors[0][i].score > successors[0][i+1].score and
                             successors[0][i].score != -float('inf'))
                            for i in range(len(successors[0])-1)))

        # reduce is valid after SHIFT
        beam = [[simulate(['NT(S)', 'NT(NP)', 'NT(NP)', 'SHIFT'])],  # can reduce
                [simulate(['NT(NP)', 'NT(NP)', 'SHIFT']),  # can reduce
                 simulate(['NT(NP)', 'SHIFT', 'NT(VP)'])]]  # cannot reduce
        successors, forced_completion_successors = model.get_successors(x, 0, beam, 12, 2)
        self.assertEqual([len(s) for s in forced_completion_successors], [1, 2])
        self.assertEqual([len(s) for s in successors], [6, 6+5])

        # only reduce after shifting all
        beam = [[simulate(['NT(S)', 'SHIFT', 'SHIFT', 'SHIFT'])],  # can finish
                [simulate(['NT(NP)', 'NT(NP)', 'SHIFT', 'SHIFT', 'SHIFT']),  # cannot finish
                 simulate(['NT(NP)', 'SHIFT', 'NT(VP)', 'SHIFT', 'SHIFT'])]]  # cannot finish
        successors, forced_completion_successors = model.get_successors(x, 3, beam, 12, 2)
        self.assertEqual([len(s) for s in forced_completion_successors], [1, 0])
        self.assertEqual([len(s) for s in successors], [1, 2])

    def test_update_stack_rnn(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        initial_beam = model.initial_beam(x)

        def simulate(actions):
            return self._simulate_beam_item(
                initial_beam[0][0], actions, model.action_dict, model.w_dim, avoid_last_stack_update=True)

        items = [simulate(['NT(S)', 'NT(NP)', 'SHIFT']),
                 simulate(['NT(S)', 'NT(NP)', 'SHIFT', 'SHIFT', 'REDUCE']),
                 simulate(['NT(S)', 'NT(NP)', 'NT(VP)', 'SHIFT', 'REDUCE']),
                 simulate(['NT(S)', 'NT(NP)', 'NT(VP)'])]

        def test_state(state, pointer, len_stack, nopen_parens,
                       ncons_nts, nt_index, nt_ids):
            self.assertEqual(state.pointer, pointer)
            self.assertEqual(len(state.stack), len_stack)
            self.assertEqual(len(state.stack_trees), len_stack-1)
            self.assertEqual(state.nopen_parens, nopen_parens)
            self.assertEqual(state.ncons_nts, ncons_nts)
            self.assertEqual(state.nt_index, nt_index)
            self.assertEqual(state.nt_ids, nt_ids)

        states = [item.state for item in items]
        word_vecs = model.emb(x)
        model.update_stack_rnn(items, states, word_vecs, [0, 0, 1, 1])
        test_state(states[0], 1, 4, 2, 0, [1, 2], [0, 1])
        test_state(states[1], 2, 3, 1, 0, [1], [0])
        test_state(states[2], 1, 4, 2, 0, [1, 2], [0, 1])
        test_state(states[3], 0, 4, 3, 3, [1, 2, 3], [0, 1, 2])

    def test_beam_search(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        parses, surprisals = model.word_sync_beam_search(x, 8, 5, 1)

        self.assertEqual(len(parses), 2)
        self.assertEqual(len(parses[0]), 5)

        paths = set([tuple(parse) for parse, score in parses[0]])
        self.assertEqual(len(paths), 5)

        for parse, score in parses[0]:
            print([model.action_dict.i2a[action] for action in parse])
        print(surprisals[0])
        self.assertEqual([len(s) for s in surprisals], [3, 3])
        self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def assertTensorAlmostEqual(self, x, y):
        self.assertIsNone(assert_almost_equal(x.cpu().detach().numpy(), y.cpu().detach().numpy()))

    def _get_simple_top_down_model(self, vocab=6, w_dim=4, h_dim=6, num_layers=2):
        a_dict = TopDownActionDict(['S', 'NP', 'VP', 'PP'])
        return TopDownRNNG(a_dict, vocab=vocab, w_dim=w_dim, h_dim=h_dim, num_layers=num_layers)

    def _trees_to_actions(self, trees):
        def conv(a):
            if a[0] ==  '(':
                return 'NT({})'.format(a[1:])
            elif a == ')':
                return 'REDUCE'
            else:
                return 'SHIFT'
        return [[conv(x) for x in tree.split()] for tree in trees]

    def _simulate_beam_item(self, item, actions, action_dict, w_dim,
                            avoid_last_stack_update=False):
        def rand_new_stack_top():
            h_dim = item.state.stack[0][0][0].size(0)
            layers = len(item.state.stack[0])
            return [([torch.rand(h_dim)], [torch.rand(h_dim)]) for _ in range(layers)]
        def rand_new_tree_elem():
            return [torch.rand(w_dim)]
        def update_stack_by_rand(state):
            state.update_stack(rand_new_stack_top(), rand_new_tree_elem(), 0)

        _item = item
        for i, a in enumerate(actions):
            a_i = action_dict.a2i[a]
            succ = SuccessorItem(_item, a_i, -1.0)
            _item = succ.to_incomplete_beam_item()
            if i == len(actions) - 1 and avoid_last_stack_update:
                break  # prepare for test for update_stack_rnn

            if action_dict.is_reduce(a_i):
                _item.reduce_stack()
            update_stack_by_rand(_item.state)
            _item.do_action(action_dict)

        return _item
