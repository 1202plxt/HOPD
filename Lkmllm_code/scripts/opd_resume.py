"""Atomic rolling resume state and committed-journal rollback. No torch import on import."""
import os
from pathlib import Path


def acquire_lock(out):
    import fcntl
    handle=(out/'training.lock').open('a')
    try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('Another trainer already holds this output directory')
    return handle


def contract(protocol):
    import copy
    p=copy.deepcopy(protocol)
    p.pop('workers',None)
    p['arguments'].pop('resume',None)
    return p


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(path,state):
    import torch
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    # Persist the directory entry on supported Linux filesystems.
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def journal_offsets(out):
    return {name:(out/name).stat().st_size if (out/name).exists() else 0
            for name in ('train.jsonl','accepted_samples.jsonl','progress.jsonl')}


def rollback_journals(out,offsets):
    for name,size in offsets.items():
        path=out/name
        if not path.exists():
            if size:raise ValueError('Missing committed journal: '+name)
            continue
        if path.stat().st_size<size:raise ValueError('Journal shorter than saved checkpoint: '+name)
        with path.open('r+b') as f:f.truncate(size)


def rng_state():
    import torch,random,numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import torch,random,numpy as np
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
