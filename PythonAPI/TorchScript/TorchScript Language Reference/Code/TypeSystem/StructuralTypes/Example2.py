# This example uses collections.namedtuple syntax to define a tuple.

import torch
from typing import NamedTuple, Tuple
from collections import namedtuple

_AnnotatedNamedTuple = NamedTuple('_NamedTupleAnnotated', [('first', int), ('second', int)])
_UnannotatedNamedTuple = namedtuple('_NamedTupleAnnotated', ['first', 'second'])

def inc(x: _AnnotatedNamedTuple) -> Tuple[int, int]:
    return (x.first + 1, x.second + 1)

m = torch.jit.script(inc)
print(inc(_UnannotatedNamedTuple(1, 2)))
