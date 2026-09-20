"""Additive Time-domain Residue Hyperdimensional Computing"""

from typing import ClassVar, Self, override

import numpy as np
import numpy.typing as npt

from .hrr import HRR, Scheme

__all__ = ["TRHC"]

type ArrayF64 = npt.NDArray[np.float64]


class TRHC(HRR):
    """Time-domain Residue Hyperdimensional Computing.

    So-called, because it deals with RHC in the time-domain, as opposed to the frequency
    domain.
    """

    data: ArrayF64
    moduli: ClassVar[list[int]] = [3, 5, 7, 11]
    basis: ClassVar[list[ArrayF64]] = []

    @override
    @classmethod
    def new(cls, dim: int, scheme: Scheme = "unitary") -> Self:
        """Create a new vector-symbol.

        Args:
            dim (int): The dimensionality of the new vector-symbol.
            scheme (Scheme): Ignored. A Gaussian vector is not unitary, and so
                would leave the residue cycle as soon as it was bound to
                itself. The argument is kept only to match `HRR.new`.

        Returns:
            A new unitary TRHC vector-symbol.
        """
        return cls.unitary(dim)

    @staticmethod
    def generate_base_vector(
        rng: np.random.Generator, modulus: int, dim: int
    ) -> ArrayF64:
        """Generates an RHC base vector in the time domain.

        Args:
        -   rng (np.random.Generator): The random number generator.
        -   modulus (int): The modulus of the base vector.
        -   dim (int): The dimension of the base vector.

        Returns:
            An TRHC base vector in the time domain.
        """

        # Only the non-negative frequencies are drawn; `irfft` mirrors them
        # into the conjugate-symmetric half, which keeps the result real.
        k_choices = np.zeros(dim // 2 + 1, dtype=int)
        k_choices[0] = 0
        k_choices[1:] = rng.choice(modulus, dim // 2)

        if dim % 2 == 0:
            # The Nyquist bin is its own conjugate, so its phase must be 0 or
            # pi. Odd moduli admit no phase of pi, leaving 0 as the only choice.
            k_choices[-1] = 0

        phases = 2 * np.pi * k_choices / modulus
        return np.fft.irfft(np.exp(1j * phases), n=dim)

    @classmethod
    def generate_basis(cls, dim: int) -> None:
        """Generate a fresh basis, one base vector per modulus.

        Args:
        -   dim (int): The dimension of the base vectors.
        """
        rng = np.random.default_rng()
        cls.basis = [cls.generate_base_vector(rng, mod, dim) for mod in cls.moduli]

    @staticmethod
    def _bind_power(vec: ArrayF64, num: int) -> ArrayF64:
        """Raise a base vector to an integer binding power.

        Equivalent to binding `vec` with itself `num` times, but evaluated in the
        frequency domain so the result stays unitary for any `num`. A power of 0
        gives the binding identity.

        Args:
        -   vec (ArrayF64): The base vector.
        -   num (int): The binding power.

        Returns:
            The base vector raised to the given binding power.
        """
        return np.fft.ifft(np.fft.fft(vec) ** num).real

    @classmethod
    def number(
        cls,
        num: int,
        dim: int,
        alternative_basis: list[ArrayF64] | None = None,
    ) -> Self:
        """Create an TRHC vector from a number.

        Args:
        -   num (int): The number to convert.
        -   dim (int): The dimension of the vector.
        -   alternative_basis (list[ArrayF64] | None): An optional alternative basis to use.

        Returns:
            An TRHC vector representing the number.

        Raises:
        -   ValueError: If `alternative_basis` is provided and is empty.
        -   ValueError: If the basis dimension does not match the dimension of the vector.
        """

        if not cls.basis:
            cls.generate_basis(dim)

        basis = cls.basis

        if alternative_basis is not None and len(alternative_basis) == 0:
            raise ValueError("alternative_basis must not be empty")
        elif alternative_basis is not None and isinstance(
            alternative_basis[0], np.ndarray
        ):
            basis = alternative_basis

        if basis[0].shape[0] != dim:
            raise ValueError("basis dimension must match dim")

        rhc_num = cls._bind_power(basis[0], num)
        for i in range(1, len(basis)):
            rhc_num = cls.bind(rhc_num, cls._bind_power(basis[i], num))

        return cls(rhc_num)

    @classmethod
    def residue_add(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical addition.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the addition.
        """
        return cls.bind(x, y)

    @classmethod
    def residue_sub(cls, x: ArrayF64, y: ArrayF64) -> ArrayF64:
        """Perform TRHC arithmetical subtraction.

        Args:
        -   x (ArrayF64): The first vector.
        -   y (ArrayF64): The second vector.

        Returns:
            The result of the subtraction.
        """
        return cls.unbind(x, y)
