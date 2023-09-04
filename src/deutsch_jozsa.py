import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import math
from typing import Callable, List
from absl import app
import numpy as np

import lib.helper as helper
import lib.ops as ops
import lib.state as state

# Functions are either constant or balanced. Distinguish via strings.
exp_constant = 'constant'
exp_balanced = 'balanced'


def make_f(dim: int = 1,
           flavor: int = exp_constant) -> Callable[[List[int]], int]:
  """Return a constant or balanced function f over 2**dim bits."""

  power2 = 2**dim
  bits = np.zeros(power2, dtype=np.uint8)
  if flavor == exp_constant:
    bits[:] = int(np.random.random() < 0.5)
  else:
    bits[np.random.choice(power2, size=power2//2, replace=False)] = 1

  # In the generalization of single-bit Deutsch, the f function
  # accepts a string of bits.
  def f(bit_string: List[int]) -> int:
    """Return f(bits) for one of the 2 possible function types."""

    # pylint: disable=no-value-for-parameter
    idx = helper.bits2val(bit_string)
    return bits[idx]

  return f


def run_experiment(nbits: int, flavor: int):
  """Run full experiment for a given flavor of f()."""

  f = make_f(nbits-1, flavor)
  u = ops.OracleUf(nbits, f)

  psi = (ops.Hadamard(nbits-1)(state.zeros(nbits-1)) *
         ops.Hadamard()(state.ones(1)))
  psi = u(psi)
  psi = (ops.Hadamard(nbits-1) * ops.Identity(1))(psi)

  # Measure all of |0>. If all close to 1.0, f() is constant.
  for idx in range(nbits - 1):
    p0, _ = ops.Measure(psi, idx, tostate=0, collapse=False)
    if not math.isclose(p0, 1.0, abs_tol=1e-5):
      return exp_balanced
  return exp_constant



if __name__ == '__main__':

  for qubits in [3, 10]:
    result = run_experiment(qubits, exp_constant)
    print('\n({} qubits) Check result: f(x): {} (Expected: {})'
          .format(qubits, result, exp_constant))
    if result != exp_constant:
      raise AssertionError('Error, expected {}'.format(exp_constant))

    result = run_experiment(qubits, exp_balanced)
    print('\n({} qubits) Check result: f(x): {} (Expected: {})'
          .format(qubits, result, exp_balanced))
    if result != exp_balanced:
      raise AssertionError('Error, expected {}'.format(exp_balanced))
    print("\n------------------------------------------------------\n") 
