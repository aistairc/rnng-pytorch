import unittest
import torch

from utils import *

class TestModels(unittest.TestCase):

    def test_in_order_actions(self):

        line = '(S (S (NP (NP (DT a) (NN cat))) (VP (VBZ walks))))'
        actions = get_in_order_actions(line)

        self.assertEqual(actions, ['SHIFT', 'NT(NP)', 'SHIFT', 'REDUCE', 'NT(NP)', 'REDUCE',
                                   'NT(S)', 'SHIFT', 'NT(VP)', 'REDUCE', 'REDUCE',
                                   'NT(S)', 'REDUCE', 'FINISH'])
