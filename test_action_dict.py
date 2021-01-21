import unittest
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose
import torch

from action_dict import TopDownActionDict, InOrderActionDict

class TestModels(unittest.TestCase):

    def test_build_tree_str(self):

        action_dict = TopDownActionDict(['S', 'NP', 'VP', 'PP'])

        actions = ['NT(S)', 'NT(NP)', 'SHIFT', 'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'REDUCE', 'REDUCE']
        action_ids = action_dict.to_id(actions)

        tokens = ['the', 'dog', 'barks']
        tags = ['X', 'X', 'X']

        tree_str = action_dict.build_tree_str(action_ids, tokens, tags)
        self.assertEqual(tree_str, '(S (NP (X the) (X dog)) (VP (X barks)))')


    def test_in_order_build_tree_str(self):

        action_dict = InOrderActionDict(['S', 'NP', 'VP', 'PP'])

        actions = ['SHIFT', 'NT(NP)', 'SHIFT', 'REDUCE', 'NT(S)', 'SHIFT',
                   'NT(VP)', 'REDUCE', 'REDUCE', 'FINISH']
        action_ids = action_dict.to_id(actions)

        tokens = ['the', 'dog', 'barks']
        tags = ['X', 'X', 'X']

        tree_str = action_dict.build_tree_str(action_ids, tokens, tags)
        self.assertEqual(tree_str, '(S (NP (X the) (X dog)) (VP (X barks)))')

    def test_in_order_build_subword_tree_str(self):

        action_dict = InOrderActionDict(['S', 'NP', 'VP', 'PP'])

        actions = ['SHIFT', 'NT(NP)', 'SHIFT', 'SHIFT', 'REDUCE', 'NT(S)', 'SHIFT',
                   'SHIFT', 'NT(VP)', 'REDUCE', 'REDUCE', 'FINISH']
        action_ids = action_dict.to_id(actions)

        tokens = ['the', 'dog', 'barks']
        subword_end_mask = [True, False, True, False, True]
        tags = ['X', 'X', 'X']

        tree_str = action_dict.build_tree_str(action_ids, tokens, tags, subword_end_mask)
        self.assertEqual(tree_str, '(S (NP (X the) (X dog)) (VP (X barks)))')

