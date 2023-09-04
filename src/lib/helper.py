"""Helper functions."""

import itertools
import math
from typing import List, Iterable, Tuple
import numpy as np


def bitprod(nbits: int) -> Iterable[int]:
  """Produce the iterable cartesian of nbits {0, 1}."""

  for bits in itertools.product([0, 1], repeat=nbits):
    yield bits


def bits2val(bits: List[int]) -> int:
  """For given bits, compute the decimal integer."""

  # We assume bits are given in high to low order. For example,
  # the bits [1, 1, 0] will produce the value 6.
  return sum(v * (1 << (len(bits)-i-1)) for i, v in enumerate(bits))


def val2bits(val: int, nbits: int) -> List[int]:
  """Convert decimal integer to list of {0, 1}."""

  # We return the bits in order high to low. For example,
  # the value 6 is being returned as [1, 1, 0].
  return [int(c) for c in format(val, f'0{nbits}b')]