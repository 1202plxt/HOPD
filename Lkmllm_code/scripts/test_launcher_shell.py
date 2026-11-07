"""Execute the shell launcher against a fake interpreter; no models or GPUs."""
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
    def run_launcher(self, backend, offload='1', extra=(), explicit=False):
        with tempfile.TemporaryDirectory(prefix='opd launch ') as temporary:
            root = Path(temporary)
            assets = root/'assets' if explicit else root
            (assets/'shared_models').mkdir(parents=True)
            (assets/'Lkmllm_data').mkdir()
            code = root/'HOPD/Lkmllm_code'
            scripts = code/'scripts'
            scripts.mkdir(parents=True)
            launcher = scripts/'run_head_opd_8gpu.sh'
            launcher.write_bytes(Path(__file__).with_name('run_head_opd_8gpu.sh').read_bytes())
            fake = root/'fake-python'
            fake.write_text('#!/usr/bin/env bash\n'
                'if [[ "$1" == -c ]]; then printf "%s\\n" "$FAKE_BACKEND"; exit 0; fi\n'
                'exec "$TEST_REAL_PY" "$TEST_CAPTURE_HELPER" "$@"\n', encoding='utf-8', newline='\n')
            fake.chmod(0o755)
            capture = root/'capture.py'
            capture.write_text('import json,os,sys\nfrom pathlib import Path\n'
                'Path(os.environ["TEST_CAPTURE_JSON"]).write_text(json.dumps(dict(argv=sys.argv[1:],env=dict(os.environ),cwd=os.getcwd())),encoding="utf-8")\n',
                encoding='utf-8')
            captured = root/'captured.json'
            env = dict(os.environ, PY=fake.as_posix(), FAKE_BACKEND=backend,
                       TEST_REAL_PY=Path(sys.executable).as_posix(), TEST_CAPTURE_HELPER=capture.as_posix(),
                       TEST_CAPTURE_JSON=str(captured), CPU_OFFLOAD=offload,
                       HIP_VISIBLE_DEVICES='3,1', CUDA_VISIBLE_DEVICES='7,2',
                       ROCR_VISIBLE_DEVICES='9', HSA_VISIBLE_DEVICES='8')
            for name in ('LK_OPD_ROOT','STUDENT','TEACHER','ANNO','VIDEOS','OUT'):
                env.pop(name, None)
            if explicit:
                env['LK_OPD_ROOT'] = assets.as_posix()
            result = subprocess.run([BASH, launcher.as_posix(), *extra], cwd=root,
                                    env=env, text=True, capture_output=True, timeout=30)
            data = json.loads(captured.read_text(encoding='utf-8')) if captured.exists() else None
            if data is not None:
                data['cwd_matches_code'] = Path(data['cwd']).samefile(code)
            return result, data, str(assets), str(code)

    def test_detect_parent_assets_and_keep_global_2k_on_hip(self):
        result, data, assets, code = self.run_launcher('hip')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = data['argv']
        self.assertIn('--auto-launch', args)
        self.assertEqual(args[args.index('--max-samples')+1], '2000')
        self.assertEqual(args[args.index('--gpus-per-worker')+1], '2')
        self.assertIn('--offload-saved-tensors', args)
        self.assertEqual(Path(args[args.index('--student-path')+1]), Path(assets)/'shared_models/Qwen3-VL-8B-Instruct')
        self.assertTrue(data['cwd_matches_code'])
        self.assertEqual(data['env']['HIP_VISIBLE_DEVICES'], '3,1')
        for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES'):
            self.assertNotIn(key, data['env'])

    def test_explicit_assets_interpreter_cuda_and_cli_overrides(self):
        result, data, assets, _ = self.run_launcher('cuda', offload='0',
            extra=('--max-samples', '8', '--skip-merge'), explicit=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = data['argv']
        self.assertNotIn('--offload-saved-tensors', args)
        self.assertEqual(args[-3:], ['--max-samples', '8', '--skip-merge'])
        self.assertEqual(Path(args[args.index('--teacher-path')+1]), Path(assets)/'shared_models/Video-opd-teacher-Qwen3-VL-32B-GRPO')
        self.assertEqual(data['env']['CUDA_VISIBLE_DEVICES'], '7,2')
        self.assertNotIn('HIP_VISIBLE_DEVICES', data['env'])

    def test_invalid_offload_never_starts_training(self):
        result, data, _, _ = self.run_launcher('hip', offload='invalid')
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(data)
        self.assertIn('CPU_OFFLOAD must be 0 or 1', result.stderr)


if __name__ == '__main__':
    unittest.main()
