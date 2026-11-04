import unittest

from train_qwen3_32b_head_opd_dp4_mp2 import boundary_decision, boundary_token_groups


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(ids)


class TestBoundarySelection(unittest.TestCase):
    def test_four_cases(self):
        valid = lambda start, end: dict(valid=True, prediction=[start, end])
        cases = [((7, 24), (11, 19), (True, True)),
                 ((10, 24), (11, 19), (False, True)),
                 ((8, 18), (10, 16), (True, False)),
                 ((10, 20), (11, 19), (False, False))]
        for student, teacher, expected in cases:
            result, _ = boundary_decision(valid(*student), valid(*teacher), [10, 20], 0.)
            self.assertEqual((result['start'], result['end']), expected)

    def test_invalid_and_tolerance(self):
        valid = lambda start, end: dict(valid=True, prediction=[start, end])
        self.assertIsNone(boundary_decision(dict(valid=False), valid(10, 20), [10, 20], 0.)[0])
        result, _ = boundary_decision(valid(9, 21), valid(10, 20), [10, 20], 1.)
        self.assertFalse(result['start'] or result['end'])

    def test_generated_number_offsets(self):
        tokens = list('The event happens in 12.3 - 25.0 seconds.')
        start, end = boundary_token_groups(CharacterTokenizer(), tokens)
        self.assertEqual(''.join(tokens[i] for i in start), '12.3')
        self.assertEqual(''.join(tokens[i] for i in end), '25.0')
        with self.assertRaises(ValueError):
            boundary_token_groups(CharacterTokenizer(), list('unparseable'))


if __name__ == '__main__':
    unittest.main()

