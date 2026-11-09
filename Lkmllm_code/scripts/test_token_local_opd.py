"""CPU regressions for independent token gates and one-pass teacher scoring."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qwen_token_local_gate import completion_positions, local_gate
import train_qwen3_32b_head_opd_dp4_mp2 as train


class Chars:
    all_special_ids = [999]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i not in self.all_special_ids)


class LocalTests(unittest.TestCase):
    def test_user_example_and_termination_positions(self):
        text='The event happens in 14 - 45 seconds.'
        ids=list(map(ord,text)); tok=Chars()
        _,_,groups=completion_positions(tok,ids)
        proposals=list(ids)
        a,b=text.index('14'),text.index('45')
        for pos,char in ((a,'2'),(a+1,'8'),(b,'7'),(b+1,'3')):
            proposals[pos]=ord(char)
        indices,kept,rows=local_gate(tok,ids,groups,proposals,[24,80])
        self.assertEqual(kept,[a,a+1,b])
        self.assertEqual([r['gain_seconds'] for r in rows if r['accepted']],[10,4,30])
        self.assertNotIn(groups[0][-1],indices)
        self.assertNotIn(groups[1][-1],indices)
        self.assertIn(0,indices)
        self.assertEqual(len(indices),len(ids)-sum(map(len,groups))+len(kept))

    def test_multidigit_token_and_digit_count_limitation(self):
        class Pieces:
            all_special_ids=[]
            chunks=['The event happens in ','6',' - ','45',' seconds.','16','1']
            def decode(self,ids,**kwargs): return ''.join(self.chunks[i] for i in ids)
        tok=Pieces();ids=[0,1,2,3,4];_,_,groups=completion_positions(tok,ids)
        proposals=list(ids);proposals[1]=5
        _,kept,_=local_gate(tok,ids,groups,proposals,[16,80])
        self.assertIn(1,kept)
        proposals[1]=6
        _,kept,_=local_gate(tok,ids,groups,proposals,[16,80])
        self.assertNotIn(1,kept)

    def test_invalid_special_and_nonpositive_gain(self):
        text='The event happens in 14 - 45 seconds.'
        ids=list(map(ord,text));tok=Chars();_,_,groups=completion_positions(tok,ids)
        for proposed in (ord('X'),999):
            top=list(ids);top[text.index('14')]=proposed
            _,kept,rows=local_gate(tok,ids,groups,top,[24,80])
            self.assertNotIn(text.index('14'),kept)
        _,kept,_=local_gate(tok,ids,groups,ids,[24,80])
        self.assertEqual(kept,[])

    def test_shared_boundary_token_rejected(self):
        class Fused:
            def decode(self,ids,**kwargs):
                return ''.join(['The event happens in ','14 - 45',' seconds.'][i] for i in ids)
        with self.assertRaisesRegex(ValueError,'Shared'):
            completion_positions(Fused(),[0,1,2])

    def test_default_script_is_global_2k_and_top10(self):
        a=train.parser().parse_args(['--student-path','s','--teacher-path','t','--anno-json','a','--video-dir','v','--output-dir','o'])
        self.assertEqual(a.max_samples,2000)
        shell=(Path(train.__file__).parent/'run_head_opd_8gpu.sh').read_text()
        self.assertIn('--max-samples 1000',shell)
        self.assertIn('--pipeline --gpu-memory-gib 48',shell)
        self.assertIn('--end-direction-ratio 2 3',shell)
        self.assertIn('--token-local-gate',shell)
        self.assertNotIn('--boundary-selective',shell)

    def test_hip_visibility_drops_conflicting_masks(self):
        result=train.device_environment(dict(HIP_VISIBLE_DEVICES='0,1',CUDA_VISIBLE_DEVICES='5',ROCR_VISIBLE_DEVICES='2',HSA_VISIBLE_DEVICES='3',OTHER='keep'),'HIP_VISIBLE_DEVICES','1')
        self.assertEqual(result,dict(HIP_VISIBLE_DEVICES='1',OTHER='keep'))

    def test_hip_auto_launcher(self):
        fake=SimpleNamespace(version=SimpleNamespace(cuda=None,hip='6.3'),cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:2))
        with patch.dict(sys.modules,{'torch':fake}),patch.dict(os.environ,{'HIP_VISIBLE_DEVICES':'0,1','CUDA_VISIBLE_DEVICES':'5'}),patch.object(train.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run,contextlib.redirect_stdout(io.StringIO()):
            train.launch_auto(SimpleNamespace(gpus_per_worker=2),['--auto-launch','--token-local-gate'])
        env=run.call_args.kwargs['env']
        self.assertEqual(env['HIP_VISIBLE_DEVICES'],'0,1')
        self.assertNotIn('CUDA_VISIBLE_DEVICES',env)

    def test_candidate_scores_teacher_once_without_teacher_generation(self):
        class Tokens:
            shape=(1,5)
            def __getitem__(self,key):return self
            def tolist(self):return [0,1,2,3,4]
        class Pieces:
            all_special_ids=[]
            chunks=['The event happens in ','14',' - ','45',' seconds.','24','75']
            def decode(self,ids,**kwargs):return ''.join(self.chunks[i] for i in ids)
        class TeacherLogits:
            def __getitem__(self,key):return self
            def argmax(self,*args):return self
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [0,5,2,6,4]
            def float(self):return self
            def log_softmax(self,*args):return self
        loss=Mock();loss.detach.return_value=.1
        student=SimpleNamespace(train=Mock());teacher=object()
        args=SimpleNamespace(token_local_gate=True,boundary_selective=False,boundary_epsilon=0.,offload_saved_tensors=False)
        fake=SimpleNamespace(no_grad=contextlib.nullcontext,isfinite=lambda value:True)
        sg=dict(valid=True,prediction=[14,45],output='The event happens in 14 - 45 seconds.')
        with patch.dict(sys.modules,{'torch':fake}),patch.object(train.video,'training_inputs',return_value=({}, {}, {}, {},100.)),patch.object(train.video,'generate',return_value=(Tokens(),sg)) as generate,patch.object(train.video,'scoring_logits',side_effect=[TeacherLogits(),TeacherLogits()]) as score,patch('qwen_token_local_gate.topk_gate',return_value=([0,1,3],[1,3],[])) as gate,patch.object(train,'reverse_kl',return_value=loss) as kl,contextlib.redirect_stdout(io.StringIO()):
            row=train.candidate_gradient(args,student,teacher,[SimpleNamespace(tokenizer=Pieces()),None],dict(gt_start=24,gt_end=80))
        self.assertEqual(generate.call_count,1)
        self.assertIs(generate.call_args.args[0],student)
        self.assertEqual([call.args[0] for call in score.call_args_list],[teacher,student])
        gate.assert_called_once()
        self.assertTrue(row['accepted'])
        self.assertEqual(row['selected_token_indices'],[0,1,3])
        self.assertEqual(kl.call_args.args[2],[0,1,3])
        loss.backward.assert_called_once()

    def test_zero_effective_positions_skip_kl_and_backward(self):
        class Tokens:
            shape=(1,5)
            def __getitem__(self,key): return self
            def tolist(self): return [0,1,2,3,4]
        tok=SimpleNamespace(all_special_ids=[],decode=lambda ids,**kw: ''.join(
            ['The event happens in ','14',' - ','45',' seconds.'][i] for i in ids))
        class Scores:
            def __getitem__(self,key): return self
            def float(self): return self
            def log_softmax(self,*args): return self
            def cpu(self): return self
        args=SimpleNamespace(token_local_gate=True,boundary_selective=False,boundary_epsilon=0.,offload_saved_tensors=False)
        fake=SimpleNamespace(no_grad=contextlib.nullcontext,isfinite=lambda value:True)
        sg=dict(valid=True,prediction=[14,45])
        with patch.dict(sys.modules,{'torch':fake}),patch.object(train.video,'training_inputs',return_value=({}, {}, {}, {},100.)),patch.object(train.video,'generate',return_value=(Tokens(),sg)),patch.object(train.video,'scoring_logits',side_effect=[Scores(),Scores()]),patch('qwen_token_local_gate.topk_gate',return_value=([],[],[])),patch.object(train,'reverse_kl') as kl,contextlib.redirect_stdout(io.StringIO()):
            row=train.candidate_gradient(args,SimpleNamespace(train=Mock()),object(),[SimpleNamespace(tokenizer=tok),None],dict(gt_start=24,gt_end=80))
        self.assertFalse(row['accepted'])
        self.assertEqual(row['reason'],'no_top10_effective_position')
        self.assertEqual(row['selected_token_indices'],[])
        kl.assert_not_called()


if __name__=='__main__':
    unittest.main()

