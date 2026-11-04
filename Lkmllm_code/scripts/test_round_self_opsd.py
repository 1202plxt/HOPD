import math
import unittest
from train_qwen3_boundary_selective_opd import boundary_decision, boundary_token_groups, selected_indices


def prediction(s, e, **kwargs):
    return dict(valid=True, prediction=[s, e], **kwargs)


class Chunks:
    def __init__(self, chunks):
        self.chunks = chunks

    def decode(self, ids, **kwargs):
        return ''.join(self.chunks[i] for i in ids)


class BoundaryTests(unittest.TestCase):
    def test_gt_inside_is_not_boundary_correctness(self):
        d, reason = boundary_decision(prediction(9, 24), prediction(12, 19), [10, 20])
        self.assertEqual((d['start'], d['end']), (False, True))
        self.assertEqual((d['start_gain'], d['end_gain']), (-1, 3))

    def test_both_start_end_and_neither(self):
        for teacher, expected in [((11, 19), (True, True)), ((10, 30), (True, False)),
                                  ((6, 20), (False, True)), ((6, 25), (False, False))]:
            d, _ = boundary_decision(prediction(7, 24), prediction(*teacher), [10, 20])
            self.assertEqual((d['start'], d['end']), expected)

    def test_strict_epsilon_and_ties(self):
        d, _ = boundary_decision(prediction(8, 22), prediction(9, 21), [10, 20], 1.)
        self.assertFalse(d['start'] or d['end'])
        d, _ = boundary_decision(prediction(9, 21), prediction(11, 19), [10, 20])
        self.assertFalse(d['start'] or d['end'])
        for eps in (-1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                boundary_decision(prediction(1, 2), prediction(1, 2), [1, 2], eps)

    def test_iou_is_never_a_boundary_gate(self):
        d, _ = boundary_decision(prediction(9, 24, iou=.99), prediction(10, 30, iou=.01), [10, 20])
        self.assertTrue(d['start'])
        self.assertFalse(d['end'])

    def test_invalid_predictions(self):
        for bad in (dict(valid=False, prediction=None), prediction(math.nan, 20), prediction(20, 10)):
            d, _ = boundary_decision(bad, prediction(10, 20), [10, 20])
            self.assertIsNone(d)

    def test_decimal_multitoken_and_eos(self):
        tok = Chunks(['The event happens in ', '1', '2', '.', '5', ' - ', '2', '0', ' seconds.', ''])
        groups = boundary_token_groups(tok, list(range(10)))
        self.assertEqual(groups, [[1, 2, 3, 4], [6, 7]])
        self.assertEqual(selected_indices(dict(start=False, end=True), groups), [6, 7])

    def test_leading_whitespace_token(self):
        tok = Chunks(['The event happens in', ' 10', ' -', ' 20', ' seconds.'])
        self.assertEqual(boundary_token_groups(tok, list(range(5))), [[1], [3]])

    def test_shared_token_rejected(self):
        tok = Chunks(['The event happens in ', '10 - 20', ' seconds.'])
        with self.assertRaisesRegex(ValueError, 'shared_start_end_token'):
            boundary_token_groups(tok, [0, 1, 2])

    def test_mixed_text_token_rejected(self):
        tok = Chunks(['The event happens in ', '10 -', ' 20', ' seconds.'])
        with self.assertRaisesRegex(ValueError, 'mixed_number_and_text_token'):
            boundary_token_groups(tok, [0, 1, 2, 3])

    def test_unsupported_answer(self):
        for text in ('[10,20]', 'The event happens in 1e1 - 20 seconds.',
                     'Reasoning. The event happens in 10 - 20 seconds.'):
            with self.assertRaisesRegex(ValueError, 'unsupported_answer_format'):
                boundary_token_groups(Chunks([text]), [0])

    def test_unstable_decoder(self):
        class Unstable:
            def decode(self, ids, **kwargs):
                return 'wrong prefix' if len(ids) == 1 else 'The event happens in 10 - 20 seconds.'
        with self.assertRaisesRegex(ValueError, 'unstable_token_offsets'):
            boundary_token_groups(Unstable(), [0, 1])


if __name__ == '__main__':
    unittest.main()
