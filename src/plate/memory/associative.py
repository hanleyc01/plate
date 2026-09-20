"""Associative memory classes."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..vsa import VSA
from .cleanup import Cleanup, Similarity, cosine

__all__ = ["Associative", "VSAAssociative", "noise_threshold"]


def noise_threshold(dim: int, z: float = 4.0) -> float:
    """An acceptance threshold scaled to the dimensionality.

    The similarity of two unrelated vectors is distributed around `0.0` with a
    standard deviation of `1 / sqrt(dim)`, so `z` deviations above chance is
    `z / sqrt(dim)`. A fixed threshold instead grows loose as `dim` shrinks.

    Args:
        dim (int): The dimensionality of the vectors.
        z (float): Defaults to `4.0`, the number of standard deviations above
            chance a match must score.

    Returns:
        The threshold, capped at `1.0`.
    """
    return min(1.0, z / math.sqrt(dim))


@dataclass(eq=False)
class Associative[T: np.inexact[Any]]:
    """Associative memory over raw arrays.

    Keys and values are held in two `Cleanup` memories kept in lockstep, so
    that row `i` of the keys is associated with row `i` of the values. A key
    is retrieved exactly when its bytes match one already stored, and
    otherwise by the nearest key above `theta`.

    Args:
        dim (int): The dimensionality of the stored vectors.
        slots (int): The number of rows initially pre-allocated.
        dtype (npt.DTypeLike): Defaults to `np.float64`, the dtype of the
            keys and values.
        sim_function (Similarity): Defaults to `cosine`, the batched
            similarity used to compare a key against the stored keys.
        theta (float | None): Defaults to `None`, meaning `noise_threshold(dim)`.
            A key scoring at or below `theta` is treated as absent.
    """

    dim: int
    slots: int
    dtype: npt.DTypeLike = field(default=np.float64)
    sim_function: Similarity[T] = field(default=cosine)
    theta: float | None = field(default=None)
    _keys: Cleanup[T] = field(init=False, repr=False)
    _values: Cleanup[T] = field(init=False, repr=False)
    _exact: dict[bytes, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta = noise_threshold(self.dim)

        self._keys = Cleanup(self.dim, self.slots, self.dtype, self.sim_function)
        self._values = Cleanup(self.dim, self.slots, self.dtype, self.sim_function)
        self._exact = {}

    def __len__(self) -> int:
        return len(self._keys)

    def _cast(self, x: npt.NDArray[T]) -> npt.NDArray[T]:
        """Cast `x` to the memory's dtype, so that stored and queried keys
        agree byte for byte.
        """
        return np.ascontiguousarray(x, dtype=self.dtype)

    def associate(self, key: npt.NDArray[T], value: npt.NDArray[T]) -> None:
        """Directly associate a key with a value, replacing the value of an
        identical key already in memory.

        Args:
            key (npt.NDArray): The key vector.
            value (npt.NDArray): The value vector to associate with `key`.
        """
        cast = self._cast(key)
        index = self._exact.get(cast.tobytes())

        if index is None:
            _ = self._keys.memorize(cast)
            _ = self._values.memorize(self._cast(value))
            self._exact[cast.tobytes()] = len(self._keys) - 1
        else:
            self._values.replace(index, self._cast(value))

    def collides(self, key: npt.NDArray[T]) -> bool:
        """Whether `key` is close enough to a stored key to be confused with
        it, meaning it scores above `theta`.

        Args:
            key (npt.NDArray): The key vector to test.

        Returns:
            `True` if some stored key scores above `theta`.
        """
        found = self._keys.nearest(self._cast(key))

        return found is not None and found[1] > float(self.theta or 0.0)

    def deref(self, key: npt.NDArray[T]) -> npt.NDArray[T] | None:
        """Dereference a key, returning the value it is associated with.

        An exact match is preferred. Failing that, the nearest key is used,
        provided it scores above `theta`.

        Args:
            key (npt.NDArray): The key vector to dereference.

        Returns:
            A copy of the associated value, or `None` if the memory is empty
            or holds no key similar enough to `key`.
        """
        cast = self._cast(key)
        index = self._exact.get(cast.tobytes())

        if index is not None:
            return self._values.get(index)

        found = self._keys.nearest(cast)

        if found is None or found[1] <= float(self.theta or 0.0):
            return None

        return self._values.get(found[0])


@dataclass(eq=False)
class VSAAssociative[V: VSA[Any]]:
    """Associative memory over vector-symbols, wrapping `Associative`.

    The memory pulls double duty, as a store for semantic pointers to tuple
    chunks as well as function chunks. Interaction comes in two forms. The
    first is *allocation*, which mints a fresh vector-symbol and associates it
    with a trace, through `VSAAssociative.alloc`. The second is
    `VSAAssociative.associate`, which maps a given key to a trace directly.

    Retrieval is done through `VSAAssociative.deref`, named to hammer home the
    semantic pointer analogy, though it serves any two directly associated
    values.

    Args:
        vsa (type[VSA]): The VSA class of the stored vector-symbols, used for
            the memory's dtype, to mint pointers, and to wrap what is recalled.
        dim (int): The dimensionality of the vector-symbols.
        slots (int): Defaults to `100`, the number of rows initially
            pre-allocated.
        sim_function (Similarity): Defaults to `cosine`. For a VSA whose
            similarity isn't cosine, pass `pairwise(vsa.similarity)`.
        theta (float | None): Defaults to `None`, meaning `noise_threshold(dim)`.
    """

    ALLOC_ATTEMPTS: ClassVar[int] = 8
    """How often `alloc` redraws a pointer that collides before giving up."""

    vsa: type[V]
    dim: int
    slots: int = field(default=100)
    sim_function: Similarity[Any] = field(default=cosine)
    theta: float | None = field(default=None)
    _mem: Associative[Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mem = Associative(
            self.dim, self.slots, self.vsa.dtype, self.sim_function, self.theta
        )

    def __len__(self) -> int:
        return len(self._mem)

    def alloc(self, trace: V) -> V:
        """Allocate a trace into the associative memory, assigning it a fresh
        semantic pointer.

        The pointer is drawn again if it lands close enough to an existing one
        to be confused with it, which grows likely as the memory fills up
        relative to `dim`.

        Args:
            trace (VSA): The vector-symbol to store.

        Returns:
            The semantic pointer now associated with `trace`.

        Raises:
        -   RuntimeError: If no distinct pointer is found in `ALLOC_ATTEMPTS`
            draws, meaning `dim` is too small for the number stored.
        """
        for _ in range(self.ALLOC_ATTEMPTS):
            ptr: V = self.vsa.new(self.dim)

            if not self._mem.collides(ptr.data):
                self.associate(ptr, trace)
                return ptr

        raise RuntimeError(
            f"no pointer distinct from the {len(self)} stored found in "
            f"{self.ALLOC_ATTEMPTS} draws; dim={self.dim} is too small"
        )

    def associate(self, key: V, trace: V) -> None:
        """Directly associate a key with a trace.

        Args:
            key (VSA): The vector-symbol to use as a key.
            trace (VSA): The vector-symbol to associate with `key`.
        """
        self._mem.associate(key.data, trace.data)

    def deref(self, ptr: V) -> V | None:
        """Dereference a semantic pointer.

        Args:
            ptr (VSA): The semantic pointer to dereference.

        Returns:
            The vector-symbol associated with `ptr`, or `None` if the memory
            is empty or holds no pointer similar enough to `ptr`.
        """
        trace = self._mem.deref(ptr.data)

        return None if trace is None else self.vsa.from_array(trace)
