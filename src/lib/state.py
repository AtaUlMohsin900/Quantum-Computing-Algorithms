# python3
"""class State wraps a tensor as underlying representation."""

import cmath
import math
import random
from typing import List, Optional

import numpy as np

import lib.helper as helper
import lib.tensor as tensor


class State(tensor.Tensor):
  """class State represents single and multi-qubit states."""

  def __repr__(self) -> str:
    s = 'State('
    s += super().__str__().replace('\n', '\n' + ' ' * len(s))
    s += ')'
    return s

  def __str__(self) -> str:
    s = f'{self.nbits}-qubit state.'
    s += ' Tensor:\n'
    s += super().__str__()
    return s

  def dump(self, desc: Optional[str] = None, prob_only: bool = True) -> None:
    dump_state(self, desc, prob_only)

  def density(self) -> tensor.Tensor:
    return tensor.Tensor(np.outer(self, self.conj()))

  def adjoint(self) -> tensor.Tensor:
    return self.conj().transpose()

  def normalize(self) -> None:
    """Renormalize the state. Sum of squared amplitudes==1.0."""

    dprod = np.conj(self) @ self
    if dprod.is_close(0.0):
      raise AssertionError('Normalizing to zero-probability state.')
    self /= np.sqrt(np.real(dprod))

  def ampl(self, *bits) -> np.complexfloating:
    """Return amplitude for state indexed by 'bits'."""

    idx = helper.bits2val(bits)
    return self[idx]

  def prob(self, *bits) -> float:
    """Return probability for state indexed by 'bits'."""

    amplitude = self.ampl(*bits)
    return np.real(amplitude.conj() * amplitude)

  def phase(self, *bits) -> float:
    """Return phase of a state from the complex amplitude."""

    amplitude = self.ampl(*bits)
    return math.degrees(cmath.phase(amplitude))

  def maxprob(self) -> (List[float], float):
    """Find state with highest probability."""

    maxbits, maxprob = [], 0.0
    for bits in helper.bitprod(self.nbits):
      cur_prob = self.prob(*bits)
      if cur_prob > maxprob:
        maxbits, maxprob = bits, cur_prob
    return maxbits, maxprob

  def apply1(self, gate, index) -> None:
    """Apply single-qubit gate to this state."""

    # To maintain qubit ordering in this infrastructure,
    # index needs to be reversed.
    #
    index = self.nbits - index - 1
    two_q = 1 << index
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]
    for g in range(0, 1 << self.nbits, 1 << (index+1)):
      for i in range(g, g + two_q):
        t1 = g00 * self[i] + g01 * self[i + two_q]
        t2 = g10 * self[i] + g11 * self[i + two_q]
        self[i] = t1
        self[i + two_q] = t2

  def applyc(self, gate, control, target) -> None:
    """Apply a controlled 2-qubit gate via explicit indexing."""

    # To maintain qubit ordering in this infrastructure,
    # index needs to be reversed.
    qbit = self.nbits - target - 1
    two_q = 2**qbit
    control = self.nbits - control - 1
    g00 = gate[0, 0]
    g01 = gate[0, 1]
    g10 = gate[1, 0]
    g11 = gate[1, 1]
    for g in range(0, 1 << self.nbits, 1 << (qbit+1)):
      idx_base = g * (1 << self.nbits)
      for i in range(g, g + two_q):
        idx = idx_base + i
        if idx & (1 << control):
          t1 = g00 * self[i] + g01 * self[i + two_q]
          t2 = g10 * self[i] + g11 * self[i + two_q]
          self[i] = t1
          self[i + two_q] = t2


# Produce a given state for a single qubit.
# We allow specification of a global phase, even though states cannot
# be distinguished when multiplied with an arbitrary complex number, aka,
# global phase.
#
def qubit(alpha: Optional[np.complexfloating] = None,
          beta: Optional[np.complexfloating] = None) -> State:
  """Produce a given state for a single qubit."""

  if alpha is None and beta is None:
    raise ValueError('Both alpha and beta need to be specified')

  if beta is None:
    beta = math.sqrt(1.0 - np.conj(alpha) * alpha)
  if alpha is None:
    alpha = math.sqrt(1.0 - np.conj(beta) * beta)

  if not math.isclose(np.conj(alpha) * alpha +
                      np.conj(beta) * beta, 1.0):
    raise ValueError('Qubit probabilities do not sum to 1.')

  qb = np.zeros(2, dtype=tensor.tensor_type())
  qb[0] = alpha
  qb[1] = beta
  return State(qb)


# The functions zeros() and ones() produce the all-zero or all-one
# computational basis vector for `d` qubits, ie,
#     |000...0> or
#     |111...1>
#
# The result of this tensor product is
#   always [1, 0, 0, ..., 0]^T or [0, 0, 0, ..., 1]^T
#
def zeros_or_ones(d: int = 1, idx: int = 0) -> State:
  """Produce the all-0/1 basis vector for `d` qubits."""

  if d < 1:
    raise ValueError('Rank must be at least 1.')
  shape = 2**d
  t = np.zeros(shape, dtype=tensor.tensor_type())
  t[idx] = 1
  return State(t)


def zeros(d: int = 1) -> State:
  """Produce state with 'd' |0>, eg., |0000>."""
  return zeros_or_ones(d, 0)


def ones(d: int = 1) -> State:
  """Produce state with 'd' |1>, eg., |1111>."""
  return zeros_or_ones(d, 2**d - 1)


def bitstring(*bits) -> State:
  """Produce a state from a given bit sequence, eg., |0101>."""

  d = len(bits)
  if d == 0:
    raise ValueError('Rank must be at least 1.')
  for _, val in enumerate(bits):
    if val != 0 and val != 1:
      raise ValueError(f'Bits must be 0 or 1, got: {val}')
  t = np.zeros(1 << d, dtype=tensor.tensor_type())
  t[helper.bits2val(bits)] = 1
  return State(t)

# These two are used so commonly, make them constants.
zero = zeros(1)
one = ones(1)

