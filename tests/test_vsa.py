import numpy as np
import pytest

from plate.vsa import HRR, TRHC

DIM = 1024

# The residues are unique modulo the product of the moduli.
PERIOD = 3 * 5 * 7 * 11

# Unrelated vectors score around 0.0 with a deviation of 1 / sqrt(DIM), so
# pseudo-orthogonality makes that deviation the scale everything is judged
# against: five of them is still far below any substantial relation.
NOISE = 1 / np.sqrt(DIM)


def unit(index: int, dim: int = 8) -> np.ndarray:
    """The `index`th standard basis vector."""
    vector = np.zeros(dim)
    vector[index] = 1.0
    return vector


@pytest.fixture(autouse=True)
def reset_trhc_basis():
    """`TRHC.basis` is shared class state, and a basis generated at one
    dimension makes the next test at another dimension fail.
    """
    TRHC.basis = []
    yield
    TRHC.basis = []


# ================= HRR similarity =================


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        # (left, right, similarity)
        (unit(0), unit(0), 1.0),  # identical
        (unit(0), -unit(0), -1.0),  # negated, not identical
        (unit(0), unit(1), 0.0),  # orthogonal
        (unit(0), 3.0 * unit(0), 1.0),  # magnitude is divided out
        (unit(0), np.zeros(8), 0.0),  # a zero vector has no direction
    ],
)
def test_hrr_similarity_is_signed(x, y, expected):
    assert HRR.similarity(x, y) == pytest.approx(expected)


def test_hrr_similarity_of_unrelated_vectors_is_near_zero():
    np.random.seed(0)
    sims = [HRR.similarity(HRR.new(DIM).data, HRR.new(DIM).data) for _ in range(200)]

    # This is the distribution `noise_threshold` is scaled to.
    assert np.mean(sims) == pytest.approx(0.0, abs=NOISE)
    assert np.std(sims) == pytest.approx(NOISE, rel=0.2)
    assert max(abs(sim) for sim in sims) < 5 * NOISE


# ================= HRR operations =================


def test_bind_is_commutative():
    np.random.seed(0)
    x, y = HRR.new(DIM).data, HRR.new(DIM).data

    assert HRR.bind(x, y) == pytest.approx(HRR.bind(y, x))


def test_bind_is_dissimilar_to_both_of_its_operands():
    np.random.seed(0)
    x, y = HRR.new(DIM).data, HRR.new(DIM).data
    bound = HRR.bind(x, y)

    assert HRR.similarity(bound, x) == pytest.approx(0.0, abs=5 * NOISE)
    assert HRR.similarity(bound, y) == pytest.approx(0.0, abs=5 * NOISE)


def test_unbind_approximately_recovers_the_other_operand():
    """`inv` is only an approximate inverse for vectors that are not unitary,
    which recovers around 0.7 rather than 1.0 whatever the dimension.
    """
    np.random.seed(0)
    x, y = HRR.new(DIM).data, HRR.new(DIM).data
    bound = HRR.bind(x, y)

    assert HRR.similarity(HRR.unbind(bound, y), x) > 0.5
    assert HRR.similarity(HRR.unbind(bound, x), y) > 0.5


def test_unbind_exactly_recovers_a_unitary_operand():
    """TRHC vectors are unitary, so binding and unbinding are exact."""
    TRHC.generate_basis(DIM)
    x, y = TRHC.number(3, DIM).data, TRHC.number(5, DIM).data

    assert HRR.unbind(HRR.bind(x, y), y) == pytest.approx(x)


def test_bundle_is_similar_to_both_of_its_operands():
    np.random.seed(0)
    x, y = HRR.new(DIM).data, HRR.new(DIM).data
    bundled = HRR.bundle(x, y)

    # Two near-orthogonal unit vectors bundle to a resultant sitting halfway
    # between them, at cos(pi / 4) from each.
    assert HRR.similarity(bundled, x) == pytest.approx(1 / np.sqrt(2), abs=5 * NOISE)
    assert HRR.similarity(bundled, y) == pytest.approx(1 / np.sqrt(2), abs=5 * NOISE)


def test_inv_is_its_own_inverse():
    np.random.seed(0)
    x = HRR.new(DIM).data

    assert HRR.inv(HRR.inv(x)) == pytest.approx(x)


# ================= TRHC base vectors =================


@pytest.mark.parametrize("dim", [7, 8, 511, 512])
@pytest.mark.parametrize("modulus", TRHC.moduli)
def test_base_vectors_are_real_unitary_and_normalized(dim, modulus):
    rng = np.random.default_rng(0)
    base = TRHC.generate_base_vector(rng, modulus, dim)

    assert base.dtype == np.float64
    assert base.shape == (dim,)
    # Unitary under binding: every frequency has a magnitude of one, so
    # binding and unbinding are exact inverses.
    assert np.abs(np.fft.rfft(base)) == pytest.approx(1.0)
    assert np.linalg.norm(base) == pytest.approx(1.0)


@pytest.mark.parametrize("modulus", TRHC.moduli)
def test_binding_a_base_vector_repeats_with_its_modulus(modulus):
    rng = np.random.default_rng(0)
    base = TRHC.generate_base_vector(rng, modulus, 512)

    def power(n: int) -> np.ndarray:
        """`base` bound with itself `n` times."""
        return np.fft.irfft(np.fft.rfft(base) ** n, n=512)

    assert power(2 + modulus) == pytest.approx(power(2))
    assert power(modulus) == pytest.approx(power(0))


def test_generate_basis_makes_one_vector_per_modulus():
    TRHC.generate_basis(512)

    assert len(TRHC.basis) == len(TRHC.moduli)
    assert all(base.shape == (512,) for base in TRHC.basis)


# ================= TRHC numbers =================


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # (left, right) of an addition that stays inside one period
        (3, 5),
        (0, 7),
        (12, 100),
        # and one that wraps around it
        (PERIOD - 5, 10),
    ],
)
def test_residue_add_encodes_the_sum(left, right):
    total = TRHC.number((left + right) % PERIOD, DIM)
    added = TRHC.residue_add(TRHC.number(left, DIM).data, TRHC.number(right, DIM).data)

    assert HRR.similarity(added, total.data) == pytest.approx(1.0)


@pytest.mark.parametrize(("left", "right"), [(9, 4), (5, 5), (3, 10)])
def test_residue_sub_encodes_the_difference(left, right):
    difference = TRHC.number((left - right) % PERIOD, DIM)
    subtracted = TRHC.residue_sub(
        TRHC.number(left, DIM).data, TRHC.number(right, DIM).data
    )

    assert HRR.similarity(subtracted, difference.data) == pytest.approx(1.0)


def test_different_numbers_are_dissimilar():
    """Residue lobes stay within a few deviations of chance, far below the
    1.0 an encoding scores against itself.
    """
    encoded = TRHC.number(0, DIM).data
    others = [TRHC.number(k, DIM).data for k in range(1, 40)]

    assert max(abs(HRR.similarity(encoded, o)) for o in others) < 10 * NOISE


def test_numbers_repeat_once_past_the_period():
    assert HRR.similarity(
        TRHC.number(PERIOD + 6, DIM).data, TRHC.number(6, DIM).data
    ) == pytest.approx(1.0)


def test_number_rejects_a_basis_of_the_wrong_dimension():
    TRHC.generate_basis(512)

    with pytest.raises(ValueError, match="basis dimension"):
        _ = TRHC.number(3, 256)


def test_number_rejects_an_empty_alternative_basis():
    with pytest.raises(ValueError, match="must not be empty"):
        _ = TRHC.number(3, DIM, alternative_basis=[])
