import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import math
from absl import app
import numpy as np

import lib.helper as helper
import lib.ops as ops
import lib.state as state

def make_f(d: int = 3, r: int = 1):
  """Construct function that will return 1 for 'solutions' bits."""

  num_inputs = 2**d
  answers = np.zeros(num_inputs, dtype=np.int32)
  answers[r] = 1

  def func(*bits):
    return answers[helper.bits2val(*bits)]

  return func


def run_experiment(nbits, r, solutions=1) -> None:
  """Run full experiment for a given flavor of f()."""

  # Note that op_zero multiplies the diagonal elements of the operator by -1,
  # except for element [0][0] which is for control. 
  zero_projector = np.zeros((2**nbits, 2**nbits))
  zero_projector[0, 0] = 1
  op_zero = ops.Operator(zero_projector)

  # Make f and Uf
  f = make_f(nbits, r)
  uf = ops.OracleUf(nbits+1, f)

  # Build state with 1 ancilla of |1>.
  psi = state.zeros(nbits) * state.ones(1)
  for i in range(nbits + 1):
    psi.apply1(ops.Hadamard(), i)

  # The Grover operator is the combination of:
  #    - phase inversion via the u unitary
  #    - inversion about the mean (see matrix above)
  hn = ops.Hadamard(nbits)
  reflection = op_zero * 2.0 - ops.Identity(nbits)
  inversion = hn(reflection(hn)) * ops.Identity()
  grover = inversion(uf)

  # Number of Grover iterations
  iterations = int(math.pi / 4 * math.sqrt(2**nbits / solutions))

  for _ in range(iterations):
    psi = grover(psi)

  # Measurement - pick element with higher probability.
  #
  # Note: We constructed the Oracle with n+1 qubits, to allow
  # for the 'xor-ancillary'. To check the result, we need to
  # ignore this ancilla.

  # Check Matrix to see if max prob is legit 
  # print("Probs:", [psi.prob(*bits) for bits in helper.bitprod(psi.nbits)])   
  
  maxbits, maxprob = psi.maxprob()
  result = f(maxbits[:-1])
  print('\n({} qubits) Search result: f(x={}) = {}, want: {}, p: {:6.4f}'
        .format(nbits, maxbits[:-1], result, r, maxprob))
  if result != 1:
    raise AssertionError('something went wrong, measured invalid state')


def main(argv):

  run_experiment(4, 12)
  print("\n------------------------------------------------------\n") 
  run_experiment(10, 500)
  print("\n------------------------------------------------------\n") 

if __name__ == '__main__':
  app.run(main)
