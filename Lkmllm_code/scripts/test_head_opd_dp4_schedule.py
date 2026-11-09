import json
import copy
import queue
import contextlib
import io
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import train_timelens_three_group_opsd as modeling
from train_qwen3_32b_head_opd_dp4_mp2 import parser, prepare, group_devices, wave_size, main, automatic_topology, launch_auto


class TrainingTests(unittest.TestCase):
    def test_student_and_teacher_gate_both_use_greedy_generation(self):
        import sys
        from types import SimpleNamespace
        import train_qwen3_32b_head_opd_dp4_mp2 as training
        student, teacher, y = object(), object(), object()
        a = SimpleNamespace(boundary_selective=False)
        sg, tg = dict(valid=True, iou=.8), dict(valid=False, iou=0.)
        with patch.dict(sys.modules, {'torch': SimpleNamespace()}), \
             patch.object(training.video, 'training_inputs', return_value=({}, {}, {}, {}, 100.)), \
             patch.object(training.video, 'generate', side_effect=[(y, sg), (None, tg)]) as generate:
            row = training.candidate_gradient(a, student, teacher, [None, None], {})
        self.assertFalse(row['accepted'])
        self.assertEqual([call.args[0] for call in generate.call_args_list], [student, teacher])
        self.assertEqual([call.kwargs['sampled'] for call in generate.call_args_list], [False, False])

    def test_greedy_generator_clears_inherited_sampling_options(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        import probe_qwen32b_privileged_k100 as video
        class Tokens:
            shape = (1, 2)
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return self
            def numel(self): return 2
            def __getitem__(self, key): return self
            def tolist(self): return [7, 9]
        tokens = Tokens()
        model = SimpleNamespace(eval=Mock(), config=SimpleNamespace(max_position_embeddings=1000),
            generation_config=SimpleNamespace(eos_token_id=9), generate=Mock(return_value=tokens))
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, decode=lambda *a, **k:'answer'))
        a = SimpleNamespace(max_new_tokens=80)
        with patch.dict(sys.modules, {'torch': SimpleNamespace(no_grad=contextlib.nullcontext)}), \
             patch.object(video, 'move', side_effect=lambda inputs, model:inputs), \
             patch.object(modeling, 'reset_rope'), \
             patch.object(video, 'parse_prediction', return_value=dict(valid=True)):
            video.generate(model, processor, dict(input_ids=tokens), dict(gt_start=1.,gt_end=2.), 100., a)
        kwargs = model.generate.call_args.kwargs
        self.assertFalse(kwargs['do_sample'])
        self.assertEqual(kwargs['num_beams'], 1)
        for key in ('temperature', 'top_p', 'top_k'):
            self.assertIsNone(kwargs[key])

    def test_automatic_pair_counts_and_odd_gpu(self):
        for count in (2,4,6,8):
            devices = list(map(str,range(count)))
            self.assertEqual(automatic_topology(devices), (count//2,devices,[]))
        self.assertEqual(automatic_topology(['7','3','1']), (1,['7','3'],['1']))
        with self.assertRaises(ValueError):
            automatic_topology(['0'])

    def test_auto_launch_uses_current_interpreter_and_selected_gpu_labels(self):
        import sys
        from types import SimpleNamespace
        fake = SimpleNamespace(version=SimpleNamespace(cuda='12.4',hip=None),
            cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:5))
        a = SimpleNamespace(gpus_per_worker=2)
        with patch.dict(sys.modules, {'torch':fake}), \
             patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'7,3,5,1,6'}), \
             patch('train_qwen3_32b_head_opd_dp4_mp2.subprocess.run',return_value=SimpleNamespace(returncode=0)) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch_auto(a,['--auto-launch','--boundary-selective']),0)
        command = run.call_args.args[0]
        self.assertEqual(command[:3],[sys.executable,'-m','torch.distributed.run'])
        self.assertIn('--nproc_per_node=2',command)
        self.assertNotIn('--auto-launch',command)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'7,3,5,1')

    def test_gpu_groups_are_disjoint(self):
        self.assertEqual([group_devices('7,3,5,1,6,2,4,0', r, 4, 2) for r in range(4)],
                         ['7,3', '5,1', '6,2', '4,0'])
        self.assertEqual(group_devices('0,1', 0, 1, 2), '0,1')
        for visible in ('0', '0,0', '0,1,2'):
            with self.assertRaises(ValueError):
                group_devices(visible, 0, 1, 2)

    def test_global_batches_rejections_and_tail(self):
        for workers in (1, 2, 3, 4):
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
            self.assertFalse(protocol['rollout_generation']['student_do_sample'])
            self.assertFalse(protocol['rollout_generation']['teacher_do_sample'])
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


import sys
from types import SimpleNamespace
from opd_four_bucket import CandidateWindow, direction_has_room, physical_layout
import train_qwen3_32b_head_opd_dp4_mp2 as pipeline
import train_qwen3_32b_head_opd_dp4_mp2 as training

def samples(n=100):
    return [dict(video_path='v', query=f'{d}-{i}', gt_start=0, gt_end=d) for d in (3, 10, 30, 100) for i in range(n)]

class PipelineTests(unittest.TestCase):
    def test_memory_fraction_caps_allocator_to_48_gib(self):
        calls=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,
            get_device_properties=lambda i:SimpleNamespace(total_memory=64*1024**3),
            set_per_process_memory_fraction=lambda fraction, device:calls.append((fraction,device))))
        with patch.dict(sys.modules, {'torch':fake}):
            pipeline.pipeline_memory_limit(SimpleNamespace(gpu_memory_gib=48))
        self.assertEqual(calls,[(.75,0),(.75,1)])


    def test_pipeline_initialization_reaches_resume_lock(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'torch': SimpleNamespace()}), patch.object(training, 'pipeline_isolate'), patch.object(training, 'pipeline_check_devices'), patch.object(training.recovery, 'acquire_lock', side_effect=RuntimeError('lock reached')) as lock:
            args = SimpleNamespace(output_dir=directory)
            with self.assertRaisesRegex(RuntimeError, 'lock reached'):
                training.pipeline_run(args, {}, [], {}, None, {'train': '3'})
            lock.assert_called_once()

    def test_physical_gpu_groups_are_disjoint_and_follow_visibility_order(self):
        self.assertEqual(physical_layout('4,2,7,6'), dict(screen='4', teacher='2,7', train='6'))
        for ids in ('0,1,2', '0,0,1,2', '0,1,2,2'):
            with self.assertRaises(ValueError):
                physical_layout(ids)

    def test_pending_work_never_exceeds_ten_including_reserved_bucket_slots(self):
        window = CandidateWindow(samples(), 100, (2, 3, 4, 1), 43)
        for _ in range(10):
            self.assertIsNotNone(window.take(['early', 'late']))
        self.assertIsNone(window.take(['early', 'late']))
        task_id = next(iter(window.pending))
        window.finish(task_id, False)
        self.assertIsNotNone(window.take(['early']))
        self.assertEqual(len(window.pending), 10)

    def test_rejections_do_not_consume_final_duration_or_direction_quotas(self):
        window = CandidateWindow(samples(), 10, (2, 3, 4, 1), 43)
        counts, quotas = (dict(early=0, late=0), dict(early=4, late=6))
        attempts = 0
        while sum(counts.values()) < 10:
            while window.take(['early', 'late']) is not None:
                pass
            task_id = next(iter(window.pending))
            accepted = attempts % 3 != 0
            if accepted:
                direction = 'early' if counts['early'] < 4 else 'late'
                self.assertTrue(direction_has_room(direction, quotas, counts))
                counts[direction] += 1
            window.finish(task_id, accepted)
            attempts += 1
        self.assertEqual(window.source.accepted, dict(short=2, medium=3, long=4, very_long=1))
        self.assertEqual(counts, quotas)
        self.assertFalse(window.pending)

    def test_resume_preserves_inflight_candidates_and_next_random_draw(self):
        original = CandidateWindow(samples(), 20, (2, 3, 4, 1), 43)
        for _ in range(7):
            original.take(['early'])
        original.finish(next(iter(original.pending)), True)
        restored = CandidateWindow(samples(), 20, (2, 3, 4, 1), 43)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assertEqual(original.pending, restored.pending)
        self.assertEqual(original.take(['late']), restored.take(['late']))

    def test_small_bucket_reservations_cannot_overfill_final_quota(self):
        window = CandidateWindow(samples(), 10, (2, 3, 4, 1), 43)
        tasks = []
        while True:
            task = window.take(['early', 'late'])
            if task is None:
                break
            tasks.append(task)
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            window.finish(task['id'], True)
        self.assertIsNone(window.take(['early']))
        self.assertEqual(window.source.accepted, window.source.quotas)

    def test_exhaustion_is_bounded_and_does_not_reuse_short_medium_long_events(self):
        window = CandidateWindow(samples(5), 10, (2, 3, 4, 1), 43, max_attempts=25)
        indices = []
        for _ in range(25):
            task = window.take(['early'])
            if task is None:
                break
            if task['bucket'] != 'very_long':
                indices.append(task['index'])
            window.finish(task['id'], False)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertIsNone(window.take(['early']))

    def test_regenerated_direction_can_reject_before_teacher_submission(self):

        class Tokens:

            def __getitem__(self, index):
                return self

            def tolist(self):
                return [123]
        task = dict(id=1, sample={'gt_end': 10})
        with patch.dict(sys.modules, {'torch': SimpleNamespace()}), patch.object(pipeline.video, 'student_input_for_direction', return_value=('si', {}, 'pool', 20)), patch.object(pipeline.video, 'generate', return_value=(Tokens(), dict(valid=True, prediction=[1, 11]))), patch.object(pipeline.video, 'teacher_input_for_direction') as teacher:
            inbox = queue.Queue()
            result = pipeline.pipeline_formal_gradient(SimpleNamespace(), None, None, None, task, dict(early=4, late=6), dict(early=0, late=6), inbox, None, [], None)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['end_direction'], 'late')
        teacher.assert_not_called()
        self.assertTrue(inbox.empty())

    def test_worker_crash_propagates_instead_of_hanging(self):
        errors = queue.Queue()
        errors.put(dict(role='teacher', traceback='test failure'))
        with self.assertRaisesRegex(RuntimeError, 'teacher worker failed'):
            pipeline.pipeline_receive(queue.Queue(), [], errors, 1)

    def test_teacher_scores_formal_tokens_and_student_forward_overlaps_teacher(self):
        import qwen_token_local_gate as gate
        events = []

        class Tokens:

            def __getitem__(self, index):
                return self

            def tolist(self):
                return [111, 222]
        tokens = Tokens()

        class Loss:

            def backward(self):
                events.append('backward')

            def detach(self):
                return 1.25

        class Inbox:

            def put(self, request):
                self.request = request
                events.append('teacher_submitted')
        inbox = Inbox()

        def score(*args):
            events.append('student_forward')
            return ['student_logits']

        def reply(*args):
            events.append('teacher_received')
            return dict(kind='teacher', id=7, log_q=['teacher_logq'], seconds=1.0)
        student = SimpleNamespace(train=lambda: None)
        processor = SimpleNamespace(tokenizer=None)
        with patch.dict(sys.modules, {'torch': SimpleNamespace(isfinite=lambda loss: True)}), patch.object(pipeline.video, 'student_input_for_direction', return_value=('si', {}, 'pool', 20)), patch.object(pipeline.video, 'generate', return_value=(tokens, dict(valid=True, prediction=[1, 9]))), patch.object(pipeline.video, 'teacher_input_for_direction', return_value=('ti', {})), patch.object(pipeline.video, 'scoring_logits', side_effect=score), patch.object(pipeline, 'pipeline_receive', side_effect=reply), patch.object(gate, 'validate_seconds_answer'), patch.object(gate, 'completion_positions', return_value=(None, None, [[0], [1]])), patch.object(gate, 'topk_gate', return_value=([0, 1], [1], [])), patch.object(pipeline, 'reverse_kl', return_value=Loss()):
            row = pipeline.pipeline_formal_gradient(SimpleNamespace(offload_saved_tensors=False, worker_timeout=5, boundary_epsilon=0), student, processor, processor, dict(id=7, sample={'gt_start': 1, 'gt_end': 10}), dict(early=4, late=6), dict(early=0, late=0), inbox, None, [], None)
        self.assertTrue(row['accepted'])
        self.assertIs(inbox.request['tokens'], tokens)
        self.assertEqual(row['rollout_token_ids'], [111, 222])
        self.assertEqual(events, ['teacher_submitted', 'student_forward', 'teacher_received', 'backward'])

    def test_plan_only_runs_without_torch_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student, teacher = (root / 'step400', root / 'teacher32b')
            for folder, hidden, layers, heads in [(student, 4096, 36, 32), (teacher, 5120, 64, 40)]:
                folder.mkdir()
                (folder / 'config.json').write_text(json.dumps(dict(model_type='qwen3_vl', text_config=dict(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, head_dim=128))))
                (folder / 'model.safetensors').touch()
            (student / 'snapshot_merge_provenance.json').write_text(json.dumps(dict(selected_heads=[dict(layer=16, head=0)])))
            train = root / 'Train'
            train.mkdir()
            (train / 'v.mp4').touch()
            (train / 'data.json').write_text(json.dumps([dict(video_path='v.mp4', events=[dict(query=f'{d}-{i}', span=[0, d]) for d in (3, 10, 30, 100) for i in range(8)])]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.dict('os.environ', {'HIP_VISIBLE_DEVICES': '0,1,2,3'}):
                result = pipeline.pipeline_main(['--student-path', str(student), '--teacher-path', str(teacher), '--anno-json', str(train / 'data.json'), '--video-dir', str(train), '--output-dir', str(root / 'out'), '--token-local-gate', '--end-direction-ratio', '2', '3', '--duration-ratio', '2', '3', '4', '1', '--max-samples', '10', '--plan-only'])
            plan = json.loads(output.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual(plan['pipeline']['physical_gpus'], dict(screen='0', teacher='1,2', train='3'))
            self.assertEqual(plan['pipeline']['candidate_window'], 10)
            self.assertEqual(plan['end_direction_quotas'], dict(early=4, late=6))
            self.assertFalse((root / 'out').exists())
