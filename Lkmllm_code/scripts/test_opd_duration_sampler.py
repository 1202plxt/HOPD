import unittest
from opd_duration_sampler import AcceptedDurationSampler, DurationRounds


def pool(n=30):
    return [dict(id=f'{length}_{i}',gt_start=100.,gt_end=100.+length)
            for length in (3,10,30) for i in range(n)]


class SamplerTests(unittest.TestCase):
    def test_resume_244_preserves_rejected_bucket_rng_and_rounds(self):
        original = DurationRounds(pool(30),10,rounds=2,seed=7)
        for flag in (True,False,True,True,False):
            original.take_wave(1)
            original.finish_wave([flag])
        state = original.state_dict()
        restored = DurationRounds(pool(30),10,rounds=2,seed=7)
        restored.load_state_dict(state)
        while True:
            if original.accepted == original.quotas:
                self.assertEqual(restored.accepted,restored.quotas)
                if not original.advance_round():
                    self.assertFalse(restored.advance_round())
                    break
                self.assertTrue(restored.advance_round())
            wave1, wave2 = original.take_wave(1), restored.take_wave(1)
            self.assertEqual(wave1,wave2)
            self.assertTrue(wave1)
            flag = original.attempts % 4 != 0
            original.finish_wave([flag]); restored.finish_wave([flag])
        self.assertEqual(original.describe(),restored.describe())

    def test_resume_244_rejects_open_wave_and_inconsistent_state(self):
        sampler = DurationRounds(pool(),10)
        sampler.take_wave(1)
        with self.assertRaises(ValueError): sampler.state_dict()
        sampler.finish_wave([False])
        state = sampler.state_dict()
        state['attempts'] += 1
        with self.assertRaises(ValueError): sampler.load_state_dict(state)

    def test_two_rounds_quotas_no_reuse_continuous_batches(self):
        sampler = DurationRounds(pool(5000), 2000, rounds=2)
        seen = set()
        pending = steps = accepted = 0
        while True:
            if sampler.accepted == sampler.quotas:
                self.assertEqual(sampler.accepted, dict(short=400, medium=800, long=800))
                if not sampler.advance_round():
                    break
            rows = sampler.take_wave(min(2, 4-pending))
            self.assertTrue(rows)
            flags = []
            for row in rows:
                self.assertNotIn(row['id'], seen)
                seen.add(row['id'])
                flags.append(len(seen) % 3 != 0)
            sampler.finish_wave(flags)
            accepted += sum(flags)
            pending += sum(flags)
            if pending == 4:
                steps += 1
                pending = 0
        self.assertEqual((accepted, steps, pending), (4000, 1000, 0))
        self.assertEqual(len(sampler.completed), 2)
        self.assertEqual(sampler.describe()['global_quotas'], dict(short=800, medium=1600, long=1600))

    def test_round_cannot_advance_early_or_reuse_candidates(self):
        sampler = DurationRounds(pool(), 10, rounds=2)
        with self.assertRaises(RuntimeError):
            sampler.advance_round()
        with self.assertRaises(ValueError):
            DurationRounds(pool(4), 10, rounds=2)

    def test_two_workers_global_quota_batch_and_rejection(self):
        sampler = AcceptedDurationSampler(pool(1000), 1000, seed=13)
        seen = set()
        pending = steps = attempted = 0
        while True:
            rows = sampler.take_wave(min(2, 4-pending))
            if not rows:
                break
            flags = []
            bucket = sampler.current
            for row in rows:
                self.assertNotIn(row['id'], seen)
                seen.add(row['id'])
                attempted += 1
                flags.append(attempted % 3 != 0)
            sampler.finish_wave(flags)
            if not any(flags):
                self.assertEqual(sampler.current, bucket)
            pending += sum(flags)
            self.assertLessEqual(pending, 4)
            if pending == 4:
                steps += 1
                pending = 0
        self.assertEqual(sampler.accepted, dict(short=200, medium=400, long=400))
        self.assertEqual((steps, pending), (250, 0))

    def test_wave_tail_attempt_limit_and_feedback(self):
        sampler = AcceptedDurationSampler(pool(), 10, max_attempts=1)
        self.assertEqual(len(sampler.take_wave(2)), 1)
        with self.assertRaises(ValueError):
            sampler.finish_wave([True, False])
        sampler.finish_wave([False])
        self.assertEqual(sampler.take_wave(2), [])

    def test_accepted_quotas_and_same_bucket_retry(self):
        sampler=AcceptedDurationSampler(pool(),10,seed=7)
        seen=[];previous_bucket=None;rejected=False
        for sample in sampler:
            if rejected:self.assertEqual(sampler.current,previous_bucket)
            previous_bucket=sampler.current
            seen.append(sample['id'])
            rejected=len(seen)%3==1
            if not rejected:sampler.accept()
        self.assertEqual(sampler.accepted,dict(short=2,medium=4,long=4))
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(sampler.stop_reason,'target_reached')

    def test_deterministic_random_interleaving(self):
        orders=[]
        for _ in range(2):
            sampler=AcceptedDurationSampler(pool(),20,seed=42)
            order=[]
            for sample in sampler:order.append(sample['id']);sampler.accept()
            orders.append(order)
        self.assertEqual(*orders)
        self.assertGreater(len({s.split('_')[0] for s in orders[0][:8]}),1)

    def test_exhaustion_does_not_relax_quota_or_repeat(self):
        sampler=AcceptedDurationSampler(pool(4),10,seed=3)
        samples=list(sampler)
        self.assertEqual(len(samples),4)
        self.assertTrue(sampler.stop_reason.startswith('exhausted_'))
        self.assertEqual(sum(sampler.accepted.values()),0)

    def test_global_attempt_limit(self):
        sampler=AcceptedDurationSampler(pool(),10,max_attempts=3)
        self.assertEqual(len(list(sampler)),3)
        self.assertEqual(sampler.stop_reason,'attempt_limit')

    def test_rounding_and_zero_weight_bucket(self):
        sampler=AcceptedDurationSampler(pool(),8)
        self.assertEqual(sum(sampler.quotas.values()),8)
        sampler=AcceptedDurationSampler(pool(),10,ratio=(0,1,0))
        for sample in sampler:
            self.assertEqual(sampler.current,'medium');sampler.accept()
        self.assertEqual(sampler.accepted,dict(short=0,medium=10,long=0))

    def test_insufficient_candidates_fail_before_model_loading(self):
        with self.assertRaisesRegex(ValueError,'candidates'):
            AcceptedDurationSampler(pool(1),10)


if __name__=='__main__':unittest.main()
