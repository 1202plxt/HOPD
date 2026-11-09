"""Four GT-duration buckets, explicit scarce-bucket reuse, serializable pending waves."""
import random
import math


class FourDurationRounds:
    names = ('short','medium','long','very_long')

    def __init__(self,samples,target,rounds=1,ratio=(2,3,4,1),seed=42,max_attempts=0):
        if len(ratio)!=4 or any(type(x)is not int or x<0 for x in ratio) or not sum(ratio) or target<1 or rounds<1:
            raise ValueError('Invalid four-bucket quotas')
        self.samples=samples
        self.rng=random.Random(seed)
        q=[target*x//sum(ratio) for x in ratio]
        order=sorted(range(4),key=lambda i:-(target*ratio[i]%sum(ratio)))
        for i in order[:target-sum(q)]:q[i]+=1
        self.quotas=dict(zip(self.names,q))
        self.orders={k:[] for k in self.names}
        for i,s in enumerate(samples):
            d=float(s['gt_end'])-float(s['gt_start'])
            if not math.isfinite(d) or d<=0:raise ValueError('Invalid GT duration')
            k=self.names[0 if d<=5 else 1 if d<=15 else 2 if d<=60 else 3]
            self.orders[k].append(i)
        for k,ids in self.orders.items():
            required=self.quotas[k]*(1 if k=='very_long' else rounds)
            if required and (not ids or (k!='very_long' and len(ids)<required)):
                raise ValueError(f'{k}: insufficient candidates for quota')
            self.rng.shuffle(ids)
        self.rounds=rounds; self.round_index=1; self.completed=[]
        self.accepted=dict.fromkeys(self.names,0);self.cursors=dict.fromkeys(self.names,0)
        self.seen={k:set() for k in self.names};self.unique_accepted={k:set() for k in self.names}
        self.current=None;self.wave=None;self.attempts=0;self.max_attempts=max_attempts;self.stop_reason=None

    def take_wave(self,limit):
        if self.wave is not None or limit<1:raise ValueError('Unfinished/invalid wave')
        remaining={k:self.quotas[k]-self.accepted[k] for k in self.names}
        if not sum(remaining.values()):self.stop_reason='target_reached';return []
        if self.current is None:
            draw=self.rng.randrange(sum(remaining.values()))
            for k in self.names:
                draw-=remaining[k]
                if draw<0:self.current=k;break
        k=self.current; ids=[]
        for _ in range(min(limit,remaining[k])):
            if self.max_attempts and self.attempts>=self.max_attempts:
                self.stop_reason='attempt_limit';break
            order=self.orders[k];cursor=self.cursors[k]
            if cursor>=len(order) and k!='very_long':
                self.stop_reason='exhausted_'+k;break
            if cursor and cursor%len(order)==0:self.rng.shuffle(order)
            index=order[cursor%len(order)];ids.append(index);self.seen[k].add(index)
            self.cursors[k]+=1;self.attempts+=1
        if not ids:return []
        self.wave=(k,ids)
        return [self.samples[i] for i in ids]

    def finish_wave(self,flags):
        if self.wave is None or len(flags)!=len(self.wave[1]):raise ValueError('Wave feedback mismatch')
        k,ids=self.wave
        for i,ok in zip(ids,flags):
            if ok:self.accepted[k]+=1;self.unique_accepted[k].add(i)
        if any(flags):self.current=None
        self.wave=None

    def advance_round(self):
        if self.accepted!=self.quotas:raise ValueError('Incomplete round')
        self.completed.append(dict(round=self.round_index,accepted=dict(self.accepted)))
        if self.round_index==self.rounds:self.stop_reason='target_reached';return False
        self.round_index+=1;self.accepted=dict.fromkeys(self.names,0);self.current=None;return True

    def state_dict(self):
        return {k:v for k,v in self.__dict__.items() if k!='samples'}

    def load_state_dict(self,state):
        self.__dict__.update(state)

    def describe(self):
        exposures={k:sum(r['accepted'][k] for r in self.completed)+
            (self.accepted[k] if len(self.completed)<self.round_index else 0) for k in self.names}
        return dict(thresholds_seconds=[5,15,60],quotas=dict(self.quotas),rounds=self.rounds,
            current_round=self.round_index,accepted=dict(self.accepted),attempted=dict(self.cursors),
            candidates={k:len(v) for k,v in self.orders.items()},completed_rounds=list(self.completed),
            unique_accepted={k:len(v) for k,v in self.unique_accepted.items()},
            accepted_exposures=exposures,
            repeated_accepted={k:exposures[k]-len(self.unique_accepted[k]) for k in self.names},
            repeat_policy='Only >60s bucket cycles shuffled candidates; all other buckets never reuse',
            stop_reason=self.stop_reason)
