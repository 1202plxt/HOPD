"""CPU protocol checks; real tensor refresh checks run when PyTorch is available."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from train_qwen3_32b_head_opd_dp4_mp2 import round_candidates, sample_identity
from train_qwen3_round_self_opsd_dp4_mp2 import parser, prepare, refresh_teacher
from probe_qwen32b_privileged_k100 import reference_interval_text

try:
    import torch
except ImportError:
    torch = None


class RoundProtocolTests(unittest.TestCase):
    def test_replay_and_disjoint(self):
        samples = [dict(video_path=str(i), query='q', gt_start=1., gt_end=2.) for i in range(40)]
        first = round_candidates(samples, 42, 1, 'replay', set(), 10)
        used = {sample_identity(s) for s in first}
        second = round_candidates(samples, 42, 2, 'disjoint', used, 10)
        self.assertEqual(len(second), 10)
        self.assertTrue(used.isdisjoint({sample_identity(s) for s in second}))
        replay = round_candidates(samples, 42, 2, 'replay', used, 0)
        self.assertEqual({sample_identity(s) for s in replay}, {sample_identity(s) for s in samples})
        self.assertNotEqual(replay[:10], first)
        self.assertEqual(round_candidates(samples, 42, 2, 'disjoint',
            {sample_identity(s) for s in samples}, 10), [])

    def test_parser_and_round_limits(self):
        args = ['--student-path', 'base', '--anno-json', 'Train/a.jsonl',
                '--video-dir', 'videos', '--output-dir', 'output']
        a = parser().parse_args(args)
        self.assertIsNone(a.teacher_path)
        self.assertEqual((a.rounds, a.teacher_refresh, a.candidate_policy), (2, 'per-round', 'replay'))
        a = parser().parse_args(args + ['--rounds', '3', '--teacher-refresh', 'fixed', '--without-reference-text'])
        self.assertEqual(a.rounds, 3)
        self.assertTrue(a.without_reference_text)

    def test_reference_preserves_source_times(self):
        text = reference_interval_text(dict(gt_start=73.5, gt_end=109.25))
        self.assertIn('73.5 - 109.25 seconds', text)
        self.assertIn('original source video', text)

    def test_real_preflight_accepts_same_model_only_for_self_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, train = root/'base', root/'Train'
            base.mkdir()
            train.mkdir()
            (base/'config.json').write_text(json.dumps(dict(model_type='qwen3_vl',
                text_config=dict(num_hidden_layers=36, num_attention_heads=32, hidden_size=4096))))
            (base/'model.safetensors').write_bytes(b'identity-only-preflight-fixture')
            (train/'v.mp4').write_bytes(b'no-decode-in-preflight')
            annotation = train/'a.json'
            annotation.write_text(json.dumps([dict(video_path='v.mp4', events=[
                dict(query='q'+str(i), span=[1.,2.]) for i in range(5)])]))
            args = ['--student-path', str(base), '--anno-json', str(annotation),
                    '--video-dir', str(train), '--output-dir', str(root/'out'),
                    '--max-samples', '2', '--max-attempts', '3']
            a = parser().parse_args(args)
            protocol, samples, _ = prepare(a)
            self.assertEqual(len(samples), 3)
            self.assertEqual(a.teacher_path, str(base))
            self.assertTrue(protocol['teacher_policy']['reference_text'])
            a = parser().parse_args(args + ['--candidate-policy', 'disjoint', '--without-reference-text'])
            protocol, samples, _ = prepare(a)
            self.assertEqual(len(samples), 5)
            self.assertEqual(a.max_attempts, 3)
            self.assertFalse(protocol['teacher_policy']['reference_text'])
            self.assertFalse((root/'out').exists())
            a = parser().parse_args(args + ['--teacher-path', str(root/'other')])
            with self.assertRaises(ValueError):
                prepare(a)

    @unittest.skipIf(torch is None, 'PyTorch not installed')
    def test_round_teacher_is_copy_not_alias(self):
        def adapter(a, b):
            return SimpleNamespace(indices=torch.tensor([1, 2]),
                A=torch.nn.Parameter(torch.tensor([[a, b]], dtype=torch.float32)),
                B=torch.nn.Parameter(torch.tensor([[b], [a]], dtype=torch.float32)))
        student, teacher = {'0.q_proj': adapter(1, 2)}, {'0.q_proj': adapter(0, 0)}
        refresh_teacher(student, teacher)
        before = teacher['0.q_proj'].A.clone()
        with torch.no_grad():
            student['0.q_proj'].A.add_(3)
        self.assertTrue(torch.equal(teacher['0.q_proj'].A, before))
        self.assertFalse(teacher['0.q_proj'].A.requires_grad)
        self.assertIsNone(teacher['0.q_proj'].A.grad)
        refresh_teacher(student, teacher)
        self.assertTrue(torch.equal(teacher['0.q_proj'].A, student['0.q_proj'].A))
        self.assertNotEqual(teacher['0.q_proj'].A.data_ptr(), student['0.q_proj'].A.data_ptr())


if __name__ == '__main__':
    unittest.main()
