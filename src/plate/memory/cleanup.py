"""Cleanup memory classes."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..vsa import VSA

__all__ = ["Cleanup", "Similarity", "VSACleanup", "cosine", "pairwise"]

type Similarity[T: np.inexact[Any]] = Callable[
    [npt.NDArray[T], npt.NDArray[T]], npt.NDArray[np.float64]
]
"""Scores a query of shape `(dim,)` against every row of a matrix of shape
`(n, dim)`, returning activations of shape `(n,)`.
"""


def cosine[T: np.inexact[Any]](
    mem: npt.NDArray[T], x: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Cosine similarity between `x` and every row of `mem`, `0.0` for rows
    where either vector is zero.

    Complex vectors are compared with the real part of their inner product.
    """
    dots = np.real(mem @ np.conj(x)).astype(np.float64)
    norms = np.linalg.norm(mem, axis=1) * np.linalg.norm(x)
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)


def pairwise[T: np.inexact[Any]](
    sim: Callable[[npt.NDArray[T], npt.NDArray[T]], float],
) -> Similarity[T]:
    """Lift a pairwise similarity, such as `VSA.similarity`, into a batched
    `Similarity`. Loops in Python, so prefer a vectorized one when possible.
    """

    def batched(mem: npt.NDArray[T], x: npt.NDArray[T]) -> npt.NDArray[np.float64]:
        return np.array([sim(x, m) for m in mem], dtype=np.float64)

    return batched


@dataclass(eq=False)
class Cleanup[T: np.inexact[Any]]:
    """Cleanup memory over raw arrays.

    Args:
        dim (int): The dimensionality of the stored vectors.
        slots (int): The number of rows initially pre-allocated.
        dtype (npt.DTypeLike): Defaults to `np.float64`, the dtype of the
            memory matrix.
        sim_function (Similarity): Defaults to `cosine`, the batched
            similarity used to compare a query against the stored vectors.
    """

    dim: int
    slots: int
    dtype: npt.DTypeLike = field(default=np.float64)
    sim_function: Similarity[T] = field(default=cosine)
    _capacity: int = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)
    _mem: npt.NDArray[T] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.slots <= 0:
            raise ValueError(f"slots must be positive, got {self.slots}")

        self._capacity = self.slots
        self._size = 0
        self._mem = np.zeros(shape=(self.slots, self.dim), dtype=self.dtype)

    def __len__(self) -> int:
        return self._size

    def _check(self, x: npt.NDArray[T]) -> None:
        """Raise if `x` is not a vector of shape `(dim,)`."""
        if x.shape != (self.dim,):
            raise ValueError(f"expected a vector of shape ({self.dim},), got {x.shape}")

    def memorize(self, x: npt.NDArray[T]) -> npt.NDArray[T]:
        """Memorize a value into the cleanup memory.

        Args:
            x (npt.NDArray): A vector of shape `(dim,)`.

        Returns:
            The vector `x` passed.
        """
        self._check(x)

        if self._size >= self._capacity:
            self._mem = np.concatenate([self._mem, np.zeros_like(self._mem)], axis=0)
            self._capacity *= 2

        self._mem[self._size, :] = x
        self._size += 1

        return x

    def get(self, index: int) -> npt.NDArray[T]:
        """A copy of the stored vector at `index`.

        Args:
            index (int): The row of the stored vector.

        Returns:
            A copy of the vector stored at `index`.
        """
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range for {self._size} stored")

        return self._mem[index].copy()

    def replace(self, index: int, x: npt.NDArray[T]) -> None:
        """Overwrite the stored vector at `index`.

        Args:
            index (int): The row to overwrite.
            x (npt.NDArray): The vector of shape `(dim,)` to store there.
        """
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range for {self._size} stored")

        self._check(x)
        self._mem[index, :] = x

    def nearest(self, x: npt.NDArray[T]) -> tuple[int, float] | None:
        """Find the stored vector most similar to `x`.

        Args:
            x (npt.NDArray): A vector to compare against.

        Returns:
            The index of the most similar stored vector and its activation, or
            `None` if the memory is empty.
        """
        if self._size == 0:
            return None

        activations = self.sim_function(self._mem[: self._size], x)
        index = int(np.argmax(activations))

        return index, float(activations[index])

    def recall(self, x: npt.NDArray[T]) -> npt.NDArray[T] | None:
        """Recall a value in the cleanup memory.

        Args:
            x (npt.NDArray): A vector to recall.

        Returns:
            A copy of the most similar stored vector, or `None` if the memory
            is empty.
        """
        found = self.nearest(x)

        return None if found is None else self.get(found[0])


@dataclass(eq=False)
class VSACleanup[V: VSA[Any]]:
    """Cleanup memory over vector-symbols, wrapping `Cleanup`.

    Args:
        vsa (type[VSA]): The VSA class of the stored vector-symbols, used for
            the memory's dtype and to wrap recalled vectors.
        dim (int): The dimensionality of the vector-symbols.
        slots (int): Defaults to `100`, the number of rows initially
            pre-allocated.
        sim_function (Similarity): Defaults to `cosine`. For a VSA whose
            similarity isn't cosine, pass `pairwise(vsa.similarity)`.
    """

    vsa: type[V]
    dim: int
    slots: int = field(default=100)
    sim_function: Similarity[Any] = field(default=cosine)
    _mem: Cleanup[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mem = Cleanup(self.dim, self.slots, self.vsa.dtype, self.sim_function)

    def __len__(self) -> int:
        return len(self._mem)

    def memorize(self, x: V) -> V:
        """Memorize a vector-symbol into the cleanup memory.

        Args:
            x (VSA): A vector-symbol.

        Returns:
            The vector-symbol `x` passed.
        """
        _ = self._mem.memorize(x.data)
        return x

    def recall(self, x: V) -> V | None:
        """Recall a vector-symbol in the cleanup memory.

        Args:
            x (VSA): A vector-symbol to recall.

        Returns:
            The most similar stored vector-symbol, or `None` if the memory is
            empty.
        """
        recalled = self._mem.recall(x.data)
        return None if recalled is None else self.vsa.from_array(recalled)
