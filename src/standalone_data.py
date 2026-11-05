"""New standalone sampling adapter; uses only helpers in the supplied archive.

This generates deterministic new experiments, not historical server recipes.
"""
from paired_data import read_log, encode, random_operation, apply_operation

def read_xes(path):
    return [dict(id=key, activities=value['raw']) for key,value in read_log(path).items()]

def mutate(sequence, vocabulary, count, rng):
    current=list(sequence)
    visited={tuple(current)}
    edits=[]
    types=('skip','insert','rework','early','late')
    for step in range(count):
        for trial in range(1000):
            op=random_operation(str(rng.choice(types)),len(current),vocabulary,rng)
            if op is None:
                continue
            proposal=apply_operation(current,op)
            if proposal is None or tuple(proposal) in visited:
                continue
            visited.add(tuple(proposal));current=proposal;edits.append(op)
            break
        else:
            raise ValueError('Cannot generate a visible non-cyclic perturbation.')
    return current,edits
