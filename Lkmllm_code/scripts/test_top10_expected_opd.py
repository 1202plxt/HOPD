"""CPU checks for the numeric acceptance rule and masked loss positions."""
import contextlib
import ast
from collections import Counter
import io
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qwen_token_local_gate import topk_gate_rows, topk_gate, completion_positions, seconds_messages, SECONDS_INSTRUCTION


class Characters:
    all_special_ids = [0]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i != 0)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tok = Characters()
        self.ids = list(map(ord, 'The event happens in 14 - 30 seconds.'))
        self.pos = self.tok.decode(self.ids).index('14') + 1
        self.groups = [[self.pos], []]

    def gate(self, teacher, student, **kwargs):
        rows = {self.pos:[dict(token_id=ord(c),teacher_log_probability=math.log(t),
                               student_log_probability=math.log(student[c])) for c,t in teacher.items()]}
        return topk_gate_rows(self.tok, self.ids, self.groups, rows, [15.,30.], **kwargs)

    def test_pass_uses_same_candidates_and_preserves_ordinary_positions(self):
        indices, kept, audit = self.gate({'5':.8,'4':.1,'0':.1}, {'5':.1,'4':.8,'0':.1})
        self.assertEqual(kept,[self.pos])
        self.assertEqual(len(indices),len(self.ids))
        self.assertAlmostEqual(audit[0]['teacher_expected_error'],.6)
        self.assertAlmostEqual(audit[0]['student_expected_error'],1.3)

    def test_top1_good_but_teacher_distribution_worse_is_rejected(self):
        indices, kept, audit = self.gate({'5':.4,'0':.35,'9':.25}, {'5':.6,'0':.2,'9':.2})
        self.assertTrue(audit[0]['candidates'][0]['accepted'])
        self.assertFalse(kept)
        self.assertFalse(indices)  # Not even ordinary-only training.

    def test_teacher_better_than_sample_but_worse_than_student_distribution_rejected(self):
        _, kept, _ = self.gate({'5':.8,'4':.1,'0':.1}, {'5':.98,'4':.01,'0':.01})
        self.assertFalse(kept)

    def test_equal_or_insufficient_gain_is_rejected(self):
        _, kept, _ = self.gate({'5':.8,'4':.1,'0':.1}, {'5':.8,'4':.1,'0':.1})
        self.assertFalse(kept)
        _, kept, _ = self.gate({'5':.8,'4':.1,'0':.1}, {'5':.1,'4':.8,'0':.1},epsilon=.5)
        self.assertFalse(kept)

    def test_invalid_candidates_are_logged_with_probability_coverage(self):
        _, kept, audit = self.gate({'5':.7,':':.1,'4':.1,'0':.1}, {'5':.1,':':.1,'4':.7,'0':.1})
        self.assertTrue(kept)
        self.assertEqual(audit[0]['valid_candidates'],3)
        self.assertAlmostEqual(audit[0]['teacher_valid_mass'],.9)
        self.assertEqual(audit[0]['candidates'][1]['reason'],'invalid_replacement')

    def test_real_termination_is_in_boundary_groups(self):
        text,match,groups = completion_positions(self.tok,self.ids)
        self.assertIn(match.end(1),groups[0])

    def test_seconds_prompt_reaches_student_and_teacher_without_duplication(self):
        def builder(privileged=False):
            return [dict(role='user',content=[dict(type='video'),dict(type='text',text='query '+('GT' if privileged else ''))])]
        wrapped = seconds_messages(seconds_messages(builder))
        for privileged in (False,True):
            text = wrapped(privileged=privileged)[0]['content'][-1]['text']
            self.assertEqual(text.count(SECONDS_INSTRUCTION),1)
            self.assertEqual('GT' in text,privileged)

    def test_tensor_adapter_normalizes_full_vocab_and_gathers_same_teacher_ids(self):
        import numpy as np
        class Tensor:
            device = 'cpu'
            def __init__(self,v): self.v=np.asarray(v)
            @property
            def shape(self): return self.v.shape
            def __getitem__(self,k): return Tensor(self.v[k])
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def to(self,*a): return self
            def tolist(self): return self.v.tolist()
            def log_softmax(self,axis):
                x=self.v-self.v.max(axis=axis,keepdims=True)
                return Tensor(x-np.log(np.exp(x).sum(axis=axis,keepdims=True)))
            def topk(self,k,dim):
                ids=np.argsort(-self.v,axis=dim)[:,:k]
                return Tensor(np.take_along_axis(self.v,ids,axis=dim)),Tensor(ids)
            def gather(self,dim,ids):return Tensor(np.take_along_axis(self.v,ids.v,axis=dim))
        t=np.full((len(self.ids),128),-30.)
        s=t.copy()
        for c,p,q in [('5',.8,.1),('4',.1,.8),('0',.1,.1)]:
            t[self.pos,ord(c)]=math.log(p);s[self.pos,ord(c)]=math.log(q)
        with patch.dict(sys.modules,{'torch':SimpleNamespace(no_grad=contextlib.nullcontext)}):
            _,kept,audit=topk_gate(self.tok,self.ids,self.groups,Tensor(t),Tensor(s),[15.,30.])
        self.assertEqual(kept,[self.pos])
        self.assertEqual(audit[0]['top_k'],10)
        self.assertAlmostEqual(audit[0]['teacher_expected_error'],.6,places=7)

    def test_start_can_cross_wrong_student_end(self):
        tok=Characters(); text='The event happens in 1.0 - 3.0 seconds.'
        ids=list(map(ord,text)); pos=text.index('1.0')
        candidates={pos:[dict(token_id=ord(c),teacher_log_probability=math.log(t),student_log_probability=math.log(s))
                         for c,t,s in [('7',.9,.01),('1',.1,.99)]]}
        indices,kept,rows=topk_gate_rows(tok,ids,[[pos],[]],candidates,[75.,78.])
        self.assertEqual(kept,[pos])
        self.assertAlmostEqual(rows[0]['teacher_expected_error'],68.6)
        self.assertAlmostEqual(rows[0]['student_expected_error'],73.94)
        self.assertAlmostEqual(rows[0]['teacher_valid_mass'],1.)
        self.assertFalse(rows[0]['candidates'][0]['candidate_interval_valid'])
        self.assertEqual(ids,list(map(ord,text)))
        self.assertIn(0,indices)

    def test_end_can_cross_wrong_student_start(self):
        tok=Characters(); text='The event happens in 20 - 30 seconds.'
        ids=list(map(ord,text)); pos=text.index('30')
        candidates={pos:[dict(token_id=ord(c),teacher_log_probability=math.log(t),student_log_probability=math.log(s))
                         for c,t,s in [('1',.9,.01),('3',.1,.99)]]}
        _,kept,rows=topk_gate_rows(tok,ids,[[],[pos]],candidates,[1.,5.])
        self.assertEqual(kept,[pos])
        self.assertFalse(rows[0]['candidates'][0]['candidate_interval_valid'])

    def test_teacher_distribution_better_than_student_but_worse_than_sample(self):
        tok=Characters(); text='The event happens in 14.4 - 30 seconds.'
        ids=list(map(ord,text)); pos=text.index('14.4')+3
        candidates={pos:[dict(token_id=ord(c),teacher_log_probability=math.log(t),student_log_probability=math.log(s))
                         for c,t,s in [('4',.2,.1),('6',.8,.9)]]}
        _,kept,rows=topk_gate_rows(tok,ids,[[pos],[]],candidates,[14.,30.])
        self.assertLess(rows[0]['teacher_expected_error'],rows[0]['student_expected_error'])
        self.assertGreater(rows[0]['teacher_expected_error'],rows[0]['original_error'])
        self.assertEqual(kept,[])

    def test_harmful_top1_can_pass_only_if_whole_top10_expectation_improves(self):
        # Top1 itself is not the criterion: all valid teacher Top10 values count.
        _,kept,rows=self.gate({'9':.36,'5':.34,'4':.30},{'9':.9,'5':.05,'4':.05})
        self.assertFalse(rows[0]['accepted'])
        self.assertEqual(kept,[])  # teacher expectation 1.74 is still worse than sample error 1

    def test_top1_can_be_harmful_but_expected_error_still_improves(self):
        tok=Characters(); text='The event happens in 14.4 - 30 seconds.'
        ids=list(map(ord,text)); pos=text.index('14.4')+3
        candidates={pos:[dict(token_id=ord(c),teacher_log_probability=math.log(t),student_log_probability=math.log(s))
                         for c,t,s in [('9',.34,.90),('0',.33,.05),('1',.33,.05)]]}
        _,kept,rows=topk_gate_rows(tok,ids,[[pos],[]],candidates,[14.,30.])
        self.assertFalse(rows[0]['candidates'][0]['accepted'])
        self.assertEqual(kept,[pos])
        self.assertAlmostEqual(rows[0]['teacher_expected_error'],.339)
        self.assertAlmostEqual(rows[0]['student_expected_error'],.815)

    def test_evaluation_hip_visibility_and_seconds_prompt(self):
        from qwen_baseline_eval_dp import worker_visibility_environment
        from qwen_baseline_eval import parser
        env=worker_visibility_environment(dict(HIP_VISIBLE_DEVICES='0,1',CUDA_VISIBLE_DEVICES='5',ROCR_VISIBLE_DEVICES='2',HSA_VISIBLE_DEVICES='3'),'1',True)
        self.assertEqual(env,dict(HIP_VISIBLE_DEVICES='1'))
        a=parser().parse_args(['--model-path','m','--anno-json','a','--video-dir','v','--output-dir','o'])
        self.assertEqual(a.prompt_mode,'decimal_seconds')


if __name__=='__main__':
    unittest.main()
