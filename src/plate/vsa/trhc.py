"""Additive Time-domain Residue Hyperdimensional Computing"""

from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt

from .hrr import HRR

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

        k_choices = np.zeros(dim, dtype=int)
        k_choices[0] = 0

        half_len = dim // 2

        if dim % 2 == 0:
            k_choices[1:half_len] = rng.choice(modulus, half_len - 1)
            # The Nyquist bin is its own conjugate, so its phase must be 0 or pi.
            k_choices[half_len] = (
                0 if rng.random() > 0.5 else (modulus // 2 if modulus % 2 == 0 else 0)
            )
            k_choices[half_len + 1 :] = -k_choices[half_len - 1 : 0 : -1]
        else:
            # Odd dim has half_len free bins (1..half_len), with no Nyquist bin.
            k_choices[1 : half_len + 1] = rng.choice(modulus, half_len)
            k_choices[half_len + 1 :] = -k_choices[half_len:0:-1]

        phases = 2 * np.pi * k_choices / modulus
        z_freq = np.exp(1j * phases)
        z_time = np.fft.ifft(z_freq)
        return z_time.real

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
