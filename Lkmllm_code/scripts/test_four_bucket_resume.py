import copy
import json
import pickle
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from opd_four_bucket import FourDurationRounds
import opd_resume as recovery


def samples():
    return [dict(video_path=str(i),query=str(d),gt_start=0,gt_end=d)
            for d,n in [(3,1000),(10,1500),(30,2000),(100,37)] for i in range(n)]


class ResumeTests(unittest.TestCase):
    def test_four_buckets_two_rounds_with_rejections_and_scarce_reuse(self):
        sampler=FourDurationRounds(samples(),2000,rounds=2)
        pending=steps=accepted=0;seen=set();repeats=0
        while accepted<4000:
            if sampler.accepted==sampler.quotas:
                self.assertEqual(sampler.accepted,dict(short=400,medium=600,long=800,very_long=200))
                sampler.advance_round()
            wave=sampler.take_wave(min(2,4-pending));self.assertTrue(wave)
            flags=[]
            for s in wave:
                key=s['query'],s['video_path']
                if key in seen:
                    self.assertEqual(s['gt_end'],100);repeats+=1
                seen.add(key);flags.append(sampler.attempts%7!=0)
            sampler.finish_wave(flags);accepted+=sum(flags);pending+=sum(flags)
            if pending==4:steps+=1;pending=0
        self.assertEqual((accepted,steps,pending),(4000,1000,0))
        self.assertGreater(repeats,0)
        self.assertLessEqual(sampler.describe()['unique_accepted']['very_long'],37)

    def test_sampler_serialization_matches_uninterrupted_pending_wave_retry(self):
        a=FourDurationRounds(samples(),100,rounds=2)
        for _ in range(7):
            a.take_wave(2);a.finish_wave([True,False])
        b=FourDurationRounds(samples(),100,rounds=2)
        b.load_state_dict(pickle.loads(pickle.dumps(a.state_dict())))
        for i in range(60):
            if a.accepted==a.quotas:a.advance_round();b.advance_round()
            wa=a.take_wave(2);wb=b.take_wave(2)
            self.assertEqual(wa,wb)
            if not wa:break
            flags=[i%3!=0]*len(wa)
            a.finish_wave(flags);b.finish_wave(flags)
            self.assertEqual(a.describe(),b.describe())

    def test_atomic_commit_keeps_old_state_on_interruption_and_rolls_back_logs(self):
        def save(value,stream):pickle.dump(value,stream)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);path=out/'resume.pt'
            with patch.dict(sys.modules,{'torch':SimpleNamespace(save=save)}):
                recovery.atomic_save(path,dict(steps=3,pending=2))
                old=path.read_bytes()
                with patch.dict(sys.modules,{'torch':SimpleNamespace(save=lambda *a:(_ for _ in ()).throw(RuntimeError('power loss')))}):
                    with self.assertRaises(RuntimeError):recovery.atomic_save(path,dict(steps=4))
                self.assertEqual(path.read_bytes(),old)
            (out/'train.jsonl').write_bytes(b'committed\n')
            offsets=recovery.journal_offsets(out)
            with (out/'train.jsonl').open('ab') as f:f.write(b'uncommitted\n')
            recovery.rollback_journals(out,offsets)
            self.assertEqual((out/'train.jsonl').read_bytes(),b'committed\n')

    def test_contract_ignores_only_resume_and_runtime(self):
        p=dict(arguments=dict(resume=False,lr=2e-5),workers=['old'],distributed=dict(workers=2))
        q=copy.deepcopy(p);q['arguments']['resume']=True;q['workers']=['new']
        self.assertEqual(recovery.contract(p),recovery.contract(q))
        q['arguments']['lr']=1e-5
        self.assertNotEqual(recovery.contract(p),recovery.contract(q))

    def test_real_torch_adamw_and_pending_gradient_roundtrip(self):
        try:import torch
        except ImportError:self.skipTest('PyTorch unavailable locally')
        def run(stop=None,state=None):
            p=torch.nn.Parameter(torch.tensor([1.,2.]));o=torch.optim.AdamW([p],lr=.01)
            begin=0
            if state:
                p.data.copy_(state['p']);o.load_state_dict(state['optimizer']);p.grad=state['gradient'];begin=state['i']
            for i in range(begin,12):
                (p.square().sum()/4).backward()
                if (i+1)%4==0:o.step();o.zero_grad(set_to_none=True)
                if i+1==stop:return dict(p=p.detach().clone(),optimizer=recovery.cpu_tree(o.state_dict()),gradient=p.grad.clone(),i=i+1)
            return p.detach()
        full=run();state=run(stop=6)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';recovery.atomic_save(path,state)
            resumed=run(state=torch.load(path,weights_only=False))
        self.assertTrue(torch.equal(full,resumed))


if __name__=='__main__':unittest.main()
