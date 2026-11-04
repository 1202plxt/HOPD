"""CPU-only checks of exact accepted-batch scheduling and GPU isolation."""
import random
import unittest
from train_qwen3_32b_head_opd_dp4_mp2 import gpu_pair, wave_size


class ScheduleTests(unittest.TestCase):
    def test_pairs(self):
        visible = '7,3,5,1,6,2,4,0'
        self.assertEqual([gpu_pair(visible, r, 4) for r in range(4)],
                         ['7,3', '5,1', '6,2', '4,0'])
        for devices, world in [('0,1', 4), ('0,1,2,3,4,5,6,6', 4), (visible, 8)]:
            with self.assertRaises(ValueError):
                gpu_pair(devices, 0, world)

    def simulate(self, flags, target, batch):
        processed = accepted = pending = 0
        seen, updates, sums, values = [], [], [0.] * 4, []
        while processed < len(flags) and accepted < target:
            n = wave_size(4, batch, pending, target, accepted, len(flags)-processed)
            self.assertGreater(n, 0)
            for rank in range(n):
                index = processed + rank
                seen.append(index)
                if flags[index]:
                    accepted += 1
                    pending += 1
                    sums[rank] += index + 1
                    values.append(index + 1)
            processed += n
            self.assertLessEqual(pending, batch)
            self.assertLessEqual(accepted, target)
            if pending and (pending == batch or accepted == target or processed == len(flags)):
                # Ranks with zero accepted samples must still contribute zero.
                self.assertAlmostEqual(sum(sums)/pending, sum(values)/len(values))
                updates.append(pending)
                sums, values, pending = [0.] * 4, [], 0
        self.assertEqual(seen, list(range(processed)))
        self.assertEqual(accepted, min(target, sum(flags)))
        self.assertEqual(sum(updates), accepted)
        self.assertTrue(all(n == batch for n in updates[:-1]))
        return updates

    def test_rejections_exhaustion_and_tail(self):
        for batch in (1, 3, 4, 7):
            for target in (1, 5, 19, 1000):
                for probability in (0., .1, .5, 1.):
                    rng = random.Random(42)
                    self.simulate([rng.random() < probability for _ in range(81)], target, batch)

    def test_original_step_count(self):
        self.assertEqual(self.simulate([True]*1200, 1000, 4), [4]*250)


if __name__ == '__main__':
    unittest.main()
