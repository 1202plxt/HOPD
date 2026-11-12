"""Random interleaving with quotas counted only after accepted OPD samples."""
import math
import random


class AcceptedDurationSampler:
    names = ('short', 'medium', 'long')

    def __init__(self, samples, target, ratio=(2, 4, 4), seed=42,
                 short_max=5., medium_max=15., max_attempts=0):
        if (len(ratio) != 3 or any(type(v) is not int or v < 0 for v in ratio)
                or not sum(ratio) or target < 1 or max_attempts < 0
                or not math.isfinite(short_max) or not math.isfinite(medium_max)
                or not 0 < short_max < medium_max):
            raise ValueError('Invalid duration quotas or thresholds')
        self.rng = random.Random(seed)
        weight = sum(ratio)
        quotas = [target * v // weight for v in ratio]
        remainder_order = sorted(range(3), key=lambda i: (-(target * ratio[i] % weight), i))
        for i in remainder_order[:target - sum(quotas)]:
            quotas[i] += 1
        self.quotas = dict(zip(self.names, quotas))
        self.buckets = {name: [] for name in self.names}
        for sample in samples:
            length = float(sample['gt_end']) - float(sample['gt_start'])
            if not math.isfinite(length) or length <= 0:
                raise ValueError('Invalid GT duration')
            name = 'short' if length <= short_max else 'medium' if length <= medium_max else 'long'
            self.buckets[name].append(sample)
        for name, bucket in self.buckets.items():
            if len(bucket) < self.quotas[name]:
                raise ValueError(f'{name} has {len(bucket)} candidates for {self.quotas[name]} accepted quota')
            self.rng.shuffle(bucket)
        self.accepted = dict.fromkeys(self.names, 0)
        self.cursors = dict.fromkeys(self.names, 0)
        self.current = None
        self.attempts = 0
        self.max_attempts = max_attempts
        self.stop_reason = None
        self.thresholds = [short_max, medium_max]

    def __iter__(self):
        return self

    def __next__(self):
        remaining = {name: self.quotas[name] - self.accepted[name] for name in self.names}
        if not sum(remaining.values()):
            self.stop_reason = 'target_reached'
            raise StopIteration
        if self.max_attempts and self.attempts >= self.max_attempts:
            self.stop_reason = 'attempt_limit'
            raise StopIteration
        if self.current is None:
            draw = self.rng.randrange(sum(remaining.values()))
            for name in self.names:
                draw -= remaining[name]
                if draw < 0:
                    self.current = name
                    break
        name = self.current
        cursor = self.cursors[name]
        if cursor >= len(self.buckets[name]):
            self.stop_reason = 'exhausted_' + name
            raise StopIteration
        self.cursors[name] += 1
        self.attempts += 1
        return self.buckets[name][cursor]

    def accept(self):
        if self.current is None:
            raise RuntimeError('No pending duration sample to accept')
        self.accepted[self.current] += 1
        self.current = None

    def take_wave(self, limit):
        """Dispatch distinct candidates from one bucket without exceeding its quota."""
        if limit < 1 or getattr(self, '_wave', None) is not None:
            raise ValueError('Invalid or unfinished wave')
        try:
            first = next(self)
        except StopIteration:
            return []
        bucket = self.current
        count = min(limit, self.quotas[bucket] - self.accepted[bucket])
        rows = [first]
        for _ in range(count - 1):
            try:
                rows.append(next(self))
            except StopIteration:
                break
        self._wave = (bucket, len(rows))
        return rows

    def finish_wave(self, accepted):
        pending = getattr(self, '_wave', None)
        if pending is None or len(accepted) != pending[1] or any(type(x) is not bool for x in accepted):
            raise ValueError('Wave feedback mismatch')
        bucket, _ = pending
        self.accepted[bucket] += sum(accepted)
        if any(accepted):
            self.current = None
        self._wave = None

    def describe(self):
        return dict(criterion='GT event duration, not video duration',
            thresholds_seconds=self.thresholds, quotas=dict(self.quotas),
            candidates={name: len(bucket) for name, bucket in self.buckets.items()},
            accepted=dict(self.accepted), attempted=dict(self.cursors),
            stop_reason=self.stop_reason,
            policy='Random draw proportional to remaining accepted quotas; reject retries same bucket; no candidate reuse')


class DurationRounds(AcceptedDurationSampler):
    """Per-round accepted quotas, continuous disjoint candidate cursors."""
    def __init__(self, samples, target, rounds=1, **kwargs):
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be positive')
        super().__init__(samples, target, **kwargs)
        self.rounds = rounds
        self.round_index = 1
        self.completed = []
        for name in self.names:
            if len(self.buckets[name]) < self.quotas[name] * rounds:
                raise ValueError(f'{name}: insufficient distinct candidates for all rounds')

    def advance_round(self):
        if self.accepted != self.quotas or getattr(self, '_wave', None) is not None:
            raise RuntimeError('Cannot advance an incomplete round')
        self.completed.append(dict(round=self.round_index, accepted=dict(self.accepted),
                                   cumulative_attempted=dict(self.cursors)))
        if self.round_index == self.rounds:
            self.stop_reason = 'target_reached'
            return False
        self.round_index += 1
        self.accepted = dict.fromkeys(self.names, 0)
        self.current = None
        self.stop_reason = None
        return True

    def state_dict(self):
        if getattr(self, '_wave', None) is not None:
            raise ValueError('Cannot checkpoint an unfinished duration wave')
        return dict(kind='duration_rounds_3_v1', rng_state=self.rng.getstate(),
                    quotas=dict(self.quotas), rounds=self.rounds, round_index=self.round_index,
                    accepted=dict(self.accepted), cursors=dict(self.cursors),
                    current=self.current, attempts=self.attempts, stop_reason=self.stop_reason,
                    completed=list(self.completed))

    def load_state_dict(self, state):
        if (state.get('kind') != 'duration_rounds_3_v1' or state.get('quotas') != self.quotas
                or state.get('rounds') != self.rounds or not 1 <= state.get('round_index', 0) <= self.rounds
                or set(state.get('accepted', {})) != set(self.names)
                or set(state.get('cursors', {})) != set(self.names)):
            raise ValueError('Saved 2:4:4 sampler contract differs')
        for name in self.names:
            if not (0 <= state['accepted'][name] <= self.quotas[name]
                    and 0 <= state['cursors'][name] <= len(self.buckets[name])):
                raise ValueError('Invalid sampler counts for '+name)
        if (state.get('attempts') != sum(state['cursors'].values())
                or state.get('current') not in (None, *self.names)):
            raise ValueError('Invalid sampler cursor or current bucket')
        self.rng.setstate(state['rng_state'])
        for name in ('round_index','accepted','cursors','current','attempts','stop_reason','completed'):
            setattr(self,name,state[name])
        self._wave = None

    def describe(self):
        return dict(super().describe(), rounds=self.rounds, current_round=self.round_index,
                    completed_rounds=list(self.completed),
                    global_quotas={k: v*self.rounds for k, v in self.quotas.items()},
                    candidate_reuse=False)
