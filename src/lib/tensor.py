"""Implementation of Tensor base class."""
# This file contains the implementation of the base "tensor" class for all the
# math in this compiler/simulator.

from __future__ import annotations
import math
import numpy as np


# Bit width of complex data types, 64 or 128.
tensor_width = 64

# All math in this package will use this base type.
# Valid values can be np.complex128 or np.complex64
def tensor_type():
  """Return complex type."""

  if tensor_width == 64:
    return np.complex64
  return np.complex128


def float_accuracy(bit_width: int = 64):
  """Set complex type bit width to 64 or 128."""

  global tensor_width
  tensor_width = bit_width


class Tensor(np.ndarray):
  """Tensor is a numpy array representing a state or operator."""

  def __new__(cls, input_array) -> Tensor:
    return np.asarray(input_array, dtype=tensor_type()).view(cls)

  def __array_finalize__(self, obj) -> None:
    if obj is None: return
    # np.ndarray has complex construction patterns. Because of this,
    # if new attributes are needed, this is the place to add them, like this:
    #    self.info = getattr(obj, 'info', None)

  @property
  def nbits(self) -> int:
    return int(math.log2(self.shape[0]))

  def is_close(self, arg) -> bool:
    """Check that a 1D or 2D tensor is numerically close to arg."""

    return np.allclose(self, arg, atol=1e-6)

  def is_hermitian(self) -> bool:
    """Check if this tensor is hermitian - Udag = U."""

    if len(self.shape) != 2:
      return False
    if self.shape[0] != self.shape[1]:
      return False
    return self.is_close(np.conj(self.transpose()))

  def is_unitary(self) -> bool:
    """Check if this tensor is unitary - Udag*U = I."""

    return Tensor(np.conj(self.transpose()) @ self).is_close(
        Tensor(np.eye(self.shape[0])))

  def is_density(self) -> bool:
    """Check if this tensor is a density operator."""

    if not self.is_hermitian():
      return False
    if np.trace(self) - 1.0 > 1e-6:
      return False
    return True

  def is_pure(self) -> bool:
    """Check if this tensor describes a pure state (else it is mixed)."""

    if not self.is_density():
      raise ValueError('ispure() can only be applied to a density matrix.')

    tr_rho2 = np.real(np.trace(self @  self))
    return np.allclose(tr_rho2, 1.0)

  def is_permutation(self) -> bool:
    """Check whether a tensor is a true permutation matrix."""

    x = self
    return (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(axis=0) == 1).all() and
            (x.sum(axis=1) == 1).all() and
            ((x == 1) | (x == 0)).all())

  def kron(self, arg: Tensor) -> Tensor:
    """Return the kronecker product of this object with arg."""

    return self.__class__(np.kron(self, arg))

  def __mul__(self, arg: Tensor) -> Tensor:
    """Inline * operator maps to kronecker product."""

    return self.kron(arg)

  def kpow(self, n: int) -> Tensor:
    """Return the tensor product of this object with itself `n` times."""

    if n == 0:
      return 1.0

    t = self
    for _ in range(n - 1):
      t = np.kron(t, self)
    return self.__class__(t)
