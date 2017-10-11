import unittest

from helpers import to_nested_tuple

class TestToNestedTuple(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(to_nested_tuple(1, 2), (1, 2))
        self.assertEqual(to_nested_tuple([1, 2], [3, 4]), ((1, 2), (3, 4)))
        self.assertEqual(
            to_nested_tuple({'a': 1, 'b': [2, 3]}, {'a': 4, 'b': [5, 6]}),
            ((1, (2, 3)), (4, (5, 6))))

    def test_empty(self):
        self.assertEqual(to_nested_tuple(None, None), (None, None))
        self.assertEqual(to_nested_tuple([], []), (None, None))
        self.assertEqual(to_nested_tuple({}, {}), (None, None))
        self.assertEqual(to_nested_tuple([1, 2, []], [3, 4, []]), ((1, 2), (3, 4)))
        self.assertEqual(to_nested_tuple([1, 2, {}], [3, 4, {}]), ((1, 2), (3, 4)))
