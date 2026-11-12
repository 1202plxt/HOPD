"""Execute the real NVIDIA train/eval shell with a fake model runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASH = os.environ.get('BASH_BIN') or shutil.which('bash')


@unittest.skipUnless(BASH, 'Bash not available')
class ShellLauncherTests(unittest.TestCase):
    def run_launcher(self, fail_training=False, complete=True, missing_annotation=False, mask='3,1', resume=False):
        with tempfile.TemporaryDirectory(prefix='opd pipeline ') as temporary:
            root = Path(temporary)
            scripts = root/'HOPD/Lkmllm_code/scripts'
            scripts.mkdir(parents=True)
            for name in ('run_head_opd_8gpu.sh', 'run_head_opd_2gpu.sh'):
                (scripts/name).write_bytes(Path(__file__).with_name(name).read_bytes())
            for name in ('Qwen3-VL-8B-Instruct', 'Video-opd-teacher-Qwen3-VL-32B-GRPO'):
                (root/'shared_models'/name).mkdir(parents=True)
            for folder, annotation in [('Train/timelens-100k','timelens-100k.jsonl'),
                    ('Test/Qvhighlights','qvhighlights-timelens.json'),
                    ('Test/Charades_sta','charades-timelens.json'),
                    ('Test/Activitynet','activitynet-timelens.json')]:
                path = root/'Lkmllm_data/datasets'/folder
                path.mkdir(parents=True)
                if not missing_annotation or 'Activitynet' not in folder:
                    (path/annotation).write_text('{}', encoding='utf-8')
            (root/'Lkmllm_data/datasets/Test/Qvhighlights/qvhighlights').mkdir()
            capture = root/'capture.py'
            capture.write_text('''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['CAPTURE']).open('a') as f:
    f.write(json.dumps(dict(args=args,env=dict(os.environ)))+'\\n')
if 'scripts/train_qwen3_32b_head_opd_dp4_mp2.py' in args:
    if os.environ['FAIL_TRAIN']=='1':sys.exit(23)
    out=Path(args[args.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(training_complete=os.environ['COMPLETE']=='1',merged=True,accepted_events=4000,accepted_target=4000)))
''', encoding='utf-8')
            fake = root/'fake-python'
            fake.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c && "$2" == *"import torch"* ]]; then echo cuda; exit 0; fi
if [[ "$1" == -c ]]; then exec "$REAL_PY" "$@"; fi
exec "$REAL_PY" "$CAPTURE_HELPER" "$@"
''', encoding='utf-8', newline='\n')
            fake.chmod(0o755)
            captured = root/'calls.jsonl'
            env = dict(os.environ, PY=fake.as_posix(), REAL_PY=Path(sys.executable).as_posix(),
                CAPTURE_HELPER=capture.as_posix(), CAPTURE=str(captured),
                FAIL_TRAIN=str(int(fail_training)), COMPLETE=str(int(complete)),
                CUDA_VISIBLE_DEVICES=mask, HIP_VISIBLE_DEVICES='7,2', ROCR_VISIBLE_DEVICES='0')
            for name in ('LK_OPD_ROOT','STUDENT','TEACHER','ANNO','VIDEOS','OUT','EVAL_OUT','CPU_OFFLOAD',
                         'QV_ANNO','QV_VIDEOS','CHARADES_ANNO','CHARADES_VIDEOS','ACTIVITYNET_ANNO','ACTIVITYNET_VIDEOS'):
                env.pop(name, None)
            env.pop('RESUME',None)
            if resume:
                output=root/'Lkmllm_data/checkpoints/existing'
                output.mkdir(parents=True)
                (output/'resume.pt').write_bytes(b'fake rolling checkpoint')
                env.update(OUT=output.as_posix(),RESUME='1')
            result = subprocess.run([BASH, (scripts/'run_head_opd_2gpu.sh').as_posix()],
                cwd=root, env=env, text=True, capture_output=True, timeout=30)
            calls = [json.loads(s) for s in captured.read_text().splitlines()] if captured.exists() else []
            return result, calls

    def test_rkl_244_two_rounds_then_three_evaluations(self):
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(len(calls), 4)
        train = calls[0]['args']
        for name, value in [('max-samples','2000'), ('rounds','2'), ('distill-kl','reverse'),
                            ('max-attempts','0'), ('grad-accum','4'), ('gpus-per-worker','2')]:
            self.assertEqual(train[train.index('--'+name)+1], value)
        i = train.index('--duration-ratio')
        self.assertEqual(train[i+1:i+4], ['2','4','4'])
        i = train.index('--targets')
        self.assertEqual(train[i+1:i+4], ['q_proj','v_proj','o_proj'])
        self.assertIn('--offload-saved-tensors', train)
        for call, split in zip(calls[1:], ['Qvhighlights','Charades_sta','Activitynet']):
            args = call['args']
            self.assertIn('scripts/qwen_baseline_eval_dp.py', args)
            self.assertEqual(args[args.index('--split')+1], split)
            self.assertEqual(args[args.index('--model-path')+1], train[train.index('--output-dir')+1])
            self.assertIn('outputs/eval_results', args[args.index('--output-dir')+1])
            self.assertEqual(args[args.index('--prompt-mode')+1], 'decimal_seconds')
            self.assertEqual(args[args.index('--min-tokens')+1], '64')
            self.assertEqual(call['env']['CUDA_VISIBLE_DEVICES'], '3,1')
            self.assertNotIn('HIP_VISIBLE_DEVICES', call['env'])

    def test_failed_training_never_starts_evaluation(self):
        result, calls = self.run_launcher(fail_training=True)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(calls), 1)

    def test_four_gpu_training_resume_but_only_two_evaluation_gpus(self):
        result,calls=self.run_launcher(mask='3,1,7,5',resume=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(calls[0]['env']['CUDA_VISIBLE_DEVICES'],'3,1,7,5')
        self.assertIn('--resume',calls[0]['args'])
        for call in calls[1:]:
            self.assertEqual(call['env']['CUDA_VISIBLE_DEVICES'],'3,1')
            self.assertIn('--resume',call['args'])

    def test_partial_training_never_starts_evaluation(self):
        result, calls = self.run_launcher(complete=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)

    def test_missing_test_annotation_fails_before_training(self):
        result, calls = self.run_launcher(missing_annotation=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
