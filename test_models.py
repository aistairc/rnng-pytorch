import unittest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import torch

from models import *
from in_order_models import *
from action_dict import TopDownActionDict, InOrderActionDict

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
        w_dim = 2
        num_labels = 8
        comp = AttentionComposition(w_dim, 0.1, num_labels)

        batch_size = 3
        children, ch_lengths = self._random_children(batch_size, w_dim)
        nt_id = torch.tensor([0, 5, 2])
        stack_state = torch.rand(batch_size, w_dim)
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

    def test_top_down_state_init(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        states = model.initial_states(x)

        self.assertEqual(len(states), 2)
        self.assertEqual(len(states[0].stack), 1)
        self.assertFalse(all(state.finished() for state in states))
        initial_second_layer = states[0].stack[0][1]
        self.assertEqual(len(initial_second_layer), 2)
        self.assertEqual(initial_second_layer[0].size(0), 6)

        top_context = model.stack_top_h(states)
        self.assertEqual(top_context.size(), (2, 6))

    def test_top_down_rnng_transition(self):
        model = self._get_simple_top_down_model()
        self._test_transition_batch_2(model)

    def _test_transition_batch_2(self, model):
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        states = model.initial_states(x)

        trees = ["(S  (NP (NP 2 3 ) ) (VP 4 ) )",
                 "(NP (NP 1   ) 2 5 )"]

        actions = self._trees_to_actions(trees)
        a_dict = model.action_dict
        actions = a_dict.mk_action_tensor(actions)
        self.assertEqual(actions.cpu().numpy().tolist(),
                         [[3, 4, 4, 1, 1, 2, 2, 5, 1, 2, 2],
                          [4, 4, 1, 2, 1, 1, 2, 0, 0, 0, 0]])

        hs = []
        def update_hs():
            hs.append(model.stack_top_h(states))

        update_hs()

        word_vecs = model.emb(x)
        model.update_stack_rnn_train(states, actions[:, 0], word_vecs)
        update_hs()
        self.assertTrue(len(states[0].stack) == len(states[1].stack) == 2)
        self.assertTrue(len(states[0].stack_trees) == len(states[0].stack_trees) == 1)
        self.assertTensorAlmostEqual(states[0].stack_trees[-1], model.nt_emb(torch.LongTensor([0]))[0])
        self.assertEqual([state.nopen_parens for state in states], [1, 1])
        self.assertEqual([state.nt_index for state in states], [[1], [1]])
        self.assertEqual([state.nt_ids for state in states], [[a_dict.nt_id(3)], [a_dict.nt_id(4)]])

        model.update_stack_rnn_train(states, actions[:, 1], word_vecs)  # (NT(NP), NT(NP))
        update_hs()
        self.assertTrue(len(states[0].stack) == len(states[1].stack) == 3)
        self.assertTrue(len(states[0].stack_trees) == len(states[1].stack_trees) == 2)
        self.assertEqual([state.nopen_parens for state in states], [2, 2])
        self.assertEqual([state.nt_index for state in states], [[1, 2], [1, 2]])
        self.assertEqual([state.nt_ids for state in states],
                         [[a_dict.nt_id(3), a_dict.nt_id(4)],
                          [a_dict.nt_id(4), a_dict.nt_id(4)]])

        model.update_stack_rnn_train(states, actions[:, 2], word_vecs)  # (NT(NP), SHIFT)
        update_hs()
        self.assertTrue(len(states[0].stack) == len(states[1].stack) == 4)
        self.assertTrue(len(states[0].stack_trees) == len(states[1].stack_trees) == 3)
        self.assertEqual([state.nopen_parens for state in states], [3, 2])
        self.assertEqual([state.ncons_nts for state in states], [3, 0])
        self.assertEqual([state.nt_index for state in states], [[1, 2, 3], [1, 2]])
        self.assertEqual([state.nt_ids for state in states],
                         [[a_dict.nt_id(3), a_dict.nt_id(4), a_dict.nt_id(4)],
                          [a_dict.nt_id(4), a_dict.nt_id(4)]])

        model.update_stack_rnn_train(states, actions[:, 3], word_vecs)  # (SHIFT, REDUCE)
        update_hs()
        self.assertTrue(len(states[0].stack) == 5)
        self.assertTrue(len(states[1].stack) == 3)
        self.assertTrue(len(states[0].stack_trees) == 4)
        self.assertTrue(len(states[1].stack_trees) == 2)
        self.assertEqual([state.nopen_parens for state in states], [3, 1])
        self.assertEqual([state.ncons_nts for state in states], [0, 0])
        self.assertEqual([state.nt_index for state in states], [[1, 2, 3], [1]])
        self.assertEqual([state.nt_ids for state in states],
                         [[a_dict.nt_id(3), a_dict.nt_id(4), a_dict.nt_id(4)],
                          [a_dict.nt_id(4)]])

        model.update_stack_rnn_train(states, actions[:, 4], word_vecs)  # (SHIFT, SHIFT)
        update_hs()
        model.update_stack_rnn_train(states, actions[:, 5], word_vecs)  # (REDUCE, SHIFT)
        update_hs()
        model.update_stack_rnn_train(states, actions[:, 6], word_vecs)  # (REDUCE, REDUCE)
        update_hs()
        self.assertTrue(len(states[0].stack) == 3)
        self.assertTrue(len(states[1].stack) == 2)
        self.assertTrue(len(states[0].stack_trees) == 2)
        self.assertTrue(len(states[1].stack_trees) == 1)
        self.assertEqual([state.nopen_parens for state in states], [1, 0])
        self.assertEqual([state.ncons_nts for state in states], [0, 0])
        self.assertEqual([state.nt_index for state in states], [[1], []])
        self.assertEqual([state.nt_ids for state in states], [[a_dict.nt_id(3)], []])
        self.assertTrue(states[1].finished())

        model.update_stack_rnn_train(states, actions[:, 7], word_vecs)  # (NT(VP), <pad>)
        update_hs()
        model.update_stack_rnn_train(states, actions[:, 8], word_vecs)  # (SHIFT, <pad>)
        update_hs()
        model.update_stack_rnn_train(states, actions[:, 9], word_vecs)  # (REDUCE, <pad>)
        update_hs()
        model.update_stack_rnn_train(states, actions[:, 10], word_vecs)  # (REDUCE, <pad>)
        update_hs()

        self.assertTrue(all(state.finished() for state in states))
        action_contexts = model.stack_to_hidden(torch.stack(hs[:-1], dim=1))
        self.assertEqual(action_contexts.size()[:2], actions.size())

        a_loss, _ = model.action_loss(actions, a_dict, action_contexts)
        w_loss, _ = model.word_loss(x, actions, a_dict, action_contexts)

        self.assertEqual(a_loss.size(), (18,))

    def test_action_path(self):
        start = ActionPath()
        end = start.add_action(1, 1.0).add_action(3, 3.0).add_action(2, 5.0)
        item = BeamItem(None, end)
        self.assertEqual(item.parse_actions(), [1, 3, 2])

    def test_get_successors(self):
        def simulate(actions):
            return self._simulate_beam_item(
                initial_beam[0][0], actions, model.action_dict, model.w_dim)

        with torch.no_grad():
            model = self._get_simple_top_down_model()
            x = torch.tensor([[2, 3, 4], [1, 2, 5]])
            initial_beam = model.initial_beam(x)
            beam = initial_beam
            new_beam, forced_completions = model.get_successors(x, 0, beam, 5, 2)
            self.assertEqual(forced_completions, [0, 0])  # shift is prohibited for initial action
            self.assertEqual([len(b) for b in new_beam], [4, 4])
            self.assertTrue(all((new_beam[0][i].score > new_beam[0][i+1].score and
                                 new_beam[0][i].score != -float('inf'))
                                for i in range(len(new_beam[0])-1)))

            # shift is valid after an NT
            beam = [[simulate(['NT(S)'])],
                    [simulate(['NT(NP)', 'NT(NP)']),
                     simulate(['NT(NP)', 'NT(VP)'])]]
            new_beam, forced_completions = model.get_successors(x, 0, beam, 8, 2)
            self.assertEqual(forced_completions, [0, 2])
            self.assertEqual([len(b) for b in new_beam], [5, 4+4+2])
            self.assertTrue(all((new_beam[0][i].score > new_beam[0][i+1].score and
                                 new_beam[0][i].score != -float('inf'))
                                for i in range(len(new_beam[0])-1)))

            # reduce is valid after SHIFT
            beam = [[simulate(['NT(S)', 'NT(NP)', 'NT(NP)', 'SHIFT'])],  # can reduce
                    [simulate(['NT(NP)', 'NT(NP)', 'SHIFT']),  # can reduce
                     simulate(['NT(NP)', 'SHIFT', 'NT(VP)'])]]  # cannot reduce
            new_beam, forced_completions = model.get_successors(x, 0, beam, 12, 2)
            self.assertEqual(forced_completions, [0, 0])
            self.assertEqual([len(s) for s in new_beam], [6, 6+5])

            # only reduce after shifting all
            beam = [[simulate(['NT(S)', 'SHIFT', 'SHIFT', 'SHIFT'])],  # can finish
                    [simulate(['NT(NP)', 'NT(NP)', 'SHIFT', 'SHIFT', 'SHIFT']),  # cannot finish
                     simulate(['NT(NP)', 'SHIFT', 'NT(VP)', 'SHIFT', 'SHIFT'])]]  # cannot finish
            new_beam, forced_completions = model.get_successors(x, 3, beam, 12, 2)
            self.assertEqual(forced_completions, [0, 0])
            self.assertEqual([len(b) for b in new_beam], [1, 2])

    def test_update_stack_rnn(self):
        model = self._get_simple_top_down_model()
        x = torch.tensor([[2, 3, 4], [1, 2, 5]])
        initial_beam = model.initial_beam(x)

        def simulate(actions):
            return self._simulate_beam_item(
                initial_beam[0][0], actions, model.action_dict, model.w_dim, avoid_last_stack_update=True)

        items = [simulate(['NT(S)', 'NT(NP)', 'SHIFT', 'SHIFT']),
                 simulate(['NT(S)', 'NT(NP)', 'SHIFT', 'REDUCE']),
                 simulate(['NT(S)', 'NT(NP)', 'NT(VP)', 'SHIFT', 'REDUCE']),
                 simulate(['NT(S)', 'SHIFT', 'NT(NP)', 'NT(VP)', 'NT(VP)'])]

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
        model.update_stack_rnn_beam(items, word_vecs, [0, 0, 0, 0], 1)
        test_state(states[0], 2, 5, 2, 0, [1, 2], [0, 1])
        test_state(states[1], 1, 3, 1, 0, [1], [0])
        test_state(states[2], 1, 4, 2, 0, [1, 2], [0, 1])
        test_state(states[3], 1, 6, 4, 3, [1, 3, 4, 5], [0, 1, 2, 2])

    def test_beam_search(self):
        with torch.no_grad():
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

    def test_beam_search_delay_word_ll(self):
        with torch.no_grad():
            model = self._get_simple_top_down_model()
            x = torch.tensor([[2, 3, 4], [1, 2, 5]])
            parses, surprisals = model.word_sync_beam_search(x, 8, 5, 0, delay_word_ll=True)

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
        with torch.no_grad():
            model = self._get_simple_top_down_model()
            x = torch.tensor([[2, 3, 4], [1, 2, 5]])
            parses, surprisals = model.variable_beam_search(x, 1000)

            self.assertEqual(len(parses), 2)
            self.assertTrue(len(parses[0]) > 0)

            paths = set([tuple(parse) for parse, score in parses[0]])
            self.assertEqual(len(paths), len(parses[0]))

            for parse, score in parses[0]:
                print([model.action_dict.i2a[action] for action in parse])
            print(surprisals[0])
            self.assertEqual([len(s) for s in surprisals], [3, 3])
            self.assertTrue(all(0 < s < float('inf') for s in surprisals[0]))

    def test_beam_search_in_order(self):
        with torch.no_grad():
            model = self._get_simple_in_order_model()
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

    def _get_simple_in_order_model(self, vocab=6, w_dim=4, h_dim=6, num_layers=2):
        a_dict = InOrderActionDict(['S', 'NP', 'VP', 'PP'])
        return InOrderRNNG(a_dict, vocab=vocab, w_dim=w_dim, h_dim=h_dim, num_layers=num_layers)

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
            return [[torch.rand(h_dim), torch.rand(h_dim)] for _ in range(layers)]
        def rand_new_tree_elem():
            return torch.rand(w_dim)
        def update_stack_by_rand(state):
            state.update_stack(rand_new_stack_top(), rand_new_tree_elem())

        _item = item
        for i, a in enumerate(actions):
            a_i = action_dict.a2i[a]
            _item = _item.next_incomplete_item(a_i, -1.0)
            if i == len(actions) - 1 and avoid_last_stack_update:
                break  # prepare for test for update_stack_rnn

            if action_dict.is_reduce(a_i):
                _item.reduce_stack()
            update_stack_by_rand(_item.state)
            _item.do_action(action_dict)

        return _item
