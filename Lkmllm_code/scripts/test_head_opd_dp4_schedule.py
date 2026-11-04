import json
import contextlib
import io
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import train_timelens_three_group_opsd as modeling
from train_qwen3_32b_head_opd_dp4_mp2 import parser, prepare, group_devices, wave_size, main


class TrainingTests(unittest.TestCase):
    def test_gpu_groups_are_disjoint(self):
        self.assertEqual([group_devices('7,3,5,1,6,2,4,0', r, 4, 2) for r in range(4)],
                         ['7,3', '5,1', '6,2', '4,0'])
        self.assertEqual(group_devices('0,1', 0, 1, 2), '0,1')
        for visible in ('0', '0,0', '0,1,2'):
            with self.assertRaises(ValueError):
                group_devices(visible, 0, 1, 2)

    def test_global_batches_rejections_and_tail(self):
        for workers in (1, 4):
            for target in (1, 9, 1000):
                for rate in (0., .2, 1.):
                    rng = random.Random(42)
                    flags = [rng.random() < rate for _ in range(1200)]
                    accepted = pending = processed = 0
                    updates, seen = [], []
                    while processed < len(flags) and accepted < target:
                        count = wave_size(workers, 4, pending, target, accepted, len(flags)-processed)
                        self.assertGreater(count, 0)
                        for rank in range(count):
                            seen.append(processed+rank)
                            if flags[processed+rank]:
                                accepted += 1
                                pending += 1
                        processed += count
                        self.assertLessEqual(pending, 4)
                        self.assertLessEqual(accepted, target)
                        if pending and (pending == 4 or accepted == target or processed == len(flags)):
                            updates.append(pending)
                            pending = 0
                    self.assertEqual(seen, list(range(processed)))
                    self.assertEqual(accepted, min(target, sum(flags)))
                    self.assertEqual(sum(updates), accepted)
                    if target == 1000 and rate == 1.:
                        self.assertEqual(updates, [4]*250)

    def test_preflight_and_no_legacy_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root/'Train'
            train.mkdir()
            (train/'v.mp4').write_bytes(b'preflight-only')
            (train/'anno.json').write_text(json.dumps([dict(video_path='v.mp4', events=[
                dict(query='q'+str(i), span=[1, 2]) for i in range(8)])]))
            for name, hidden, layers in (('student', 4096, 36), ('teacher', 5120, 64)):
                base = root/name
                base.mkdir()
                (base/'config.json').write_text(json.dumps(dict(model_type='qwen3_vl', text_config=dict(
                    hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=32))))
                (base/'model.safetensors').write_bytes(b'identity-only-fixture')
            values = ['--student-path', str(root/'student'), '--teacher-path', str(root/'teacher'),
                      '--anno-json', str(train/'anno.json'), '--video-dir', str(train),
                      '--output-dir', str(root/'out'), '--max-samples', '4']
            a = parser().parse_args(values+['--boundary-selective'])
            protocol, samples = prepare(a)
            self.assertEqual(a.lr, 2e-5)
            self.assertEqual(len(samples), 8)
            self.assertEqual(protocol['method'], 'boundary_selective_opd')
            self.assertFalse(protocol['boundary_policy']['overall_iou_gate'])
            self.assertFalse((root/'out').exists())
            output = io.StringIO()
            with patch.dict(os.environ, dict(RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', CUDA_VISIBLE_DEVICES='0,1')), contextlib.redirect_stdout(output):
                self.assertEqual(main(values+['--boundary-selective', '--plan-only']), 0)
            self.assertEqual(json.loads(output.getvalue())['method'], 'boundary_selective_opd')
            self.assertFalse((root/'out').exists())
            a = parser().parse_args(values)
            self.assertEqual(prepare(a)[0]['method'], 'whole_answer_opd_baseline')
            a.teacher_path = a.student_path
            with self.assertRaises(ValueError):
                prepare(a)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed; run on server')
class TensorTests(unittest.TestCase):
    def test_only_selected_positions_receive_direct_gradient(self):
        from train_qwen3_boundary_selective_opd import reverse_kl
        student = torch.randn(1, 7, 11, requires_grad=True)
        teacher = torch.randn(1, 7, 11, requires_grad=True)
        log_q = teacher.log_softmax(-1)
        loss = reverse_kl(student, log_q, [2, 4])
        expected = (student[:, [2,4]].softmax(-1)*(student[:, [2,4]].log_softmax(-1)-log_q[:, [2,4]])).sum(-1).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[:, [0,1,3,5,6]].abs().sum().item(), 0.)
        self.assertGreater(student.grad[:, [2,4]].abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)

    def test_causal_alignment(self):
        prompt = dict(input_ids=torch.tensor([[10, 11, 12]]), attention_mask=torch.ones(1,3,dtype=torch.long))
        y = torch.tensor([[20,21,22,23]])
        scoring, count = modeling.scoring_inputs(prompt, y)
        self.assertEqual(count, 4)
        self.assertEqual(scoring['input_ids'].tolist(), [[10,11,12,20,21,22]])
        self.assertEqual(scoring['input_ids'][:, -count:].tolist(), [[12,20,21,22]])

    def test_candidate_lower_iou_can_train_only_better_boundary(self):
        from types import SimpleNamespace
        from train_qwen3_32b_head_opd_dp4_mp2 import candidate_gradient
        from test_round_self_opsd import Chunks
        tok = Chunks(['The event happens in ', '9', ' - ', '24', ' seconds.'])
        y = torch.tensor([[0,1,2,3,4]])
        student_logits = torch.randn(1,5,9,requires_grad=True)
        teacher_logits = torch.randn(1,5,9,requires_grad=True)
        student = SimpleNamespace(train=lambda:None)
        teacher = object()
        sg = dict(valid=True,prediction=[9,24],iou=.9)
        tg = dict(valid=True,prediction=[10,30],iou=.1)
        a = SimpleNamespace(boundary_selective=True,boundary_epsilon=0.,offload_saved_tensors=False)
        processors = [SimpleNamespace(tokenizer=tok), None]
        with patch('train_qwen3_32b_head_opd_dp4_mp2.video.training_inputs',return_value=({}, {}, {}, {}, 100.)), \
             patch('train_qwen3_32b_head_opd_dp4_mp2.video.generate',side_effect=[(y,sg),(y,tg)]), \
             patch('train_qwen3_32b_head_opd_dp4_mp2.video.scoring_logits',side_effect=[teacher_logits,student_logits]):
            row = candidate_gradient(a,student,teacher,processors,dict(gt_start=10.,gt_end=20.))
        self.assertTrue(row['accepted'])
        self.assertEqual(row['selected_token_indices'], [1])
        self.assertEqual(student_logits.grad[:, [0,2,3,4]].abs().sum().item(),0.)
        self.assertIsNone(teacher_logits.grad)


if __name__ == '__main__':
    unittest.main()
