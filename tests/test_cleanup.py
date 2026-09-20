import numpy as np
import pytest

from plate.memory.cleanup import Cleanup, VSACleanup, cosine, pairwise
from plate.vsa import HRR

DIM = 8


def unit(index: int, dim: int = DIM) -> np.ndarray:
    """The `index`th standard basis vector."""
    vector = np.zeros(dim)
    vector[index] = 1.0
    return vector


def vec(*values: float) -> np.ndarray:
    """A literal float vector."""
    return np.array(values, dtype=np.float64)


def negated_cosine(mem: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A similarity that ranks the least similar vector highest."""
    return -cosine(mem, x)


# ================= cosine =================


@pytest.mark.parametrize(
    ("stored", "query", "expected"),
    [
        # (row, query, score)
        (unit(0), unit(0), 1.0),  # identical
        (unit(0), unit(1), 0.0),  # orthogonal
        (-unit(0), unit(0), -1.0),  # negated
        (3.0 * unit(0), unit(0), 1.0),  # magnitude is divided out
        (unit(0), 3.0 * unit(0), 1.0),
        (np.zeros(DIM), unit(0), 0.0),  # a zero row scores zero, not NaN
        (unit(0), np.zeros(DIM), 0.0),
    ],
)
def test_cosine_scores(stored, query, expected):
    assert cosine(stored[np.newaxis, :], query) == pytest.approx(expected)


def test_cosine_returns_one_float_per_row():
    mem = np.stack([unit(0), unit(1), unit(2)])
    activations = cosine(mem, unit(1))

    assert activations.shape == (3,)
    assert activations.dtype == np.float64
    assert activations == pytest.approx([0.0, 1.0, 0.0])


def test_cosine_of_a_zero_row_does_not_warn():
    mem = np.stack([np.zeros(DIM), unit(0)])

    with np.errstate(invalid="raise", divide="raise"):
        activations = cosine(mem, unit(0))

    assert not np.isnan(activations).any()
    assert activations == pytest.approx([0.0, 1.0])


def test_cosine_compares_complex_vectors_by_their_inner_product():
    phases = np.array([0.0, 1.0, 2.0, 3.0])
    phasors = np.exp(1j * phases)
    mem = np.stack([phasors, -phasors, np.conj(phasors)])

    # Conjugating a row doubles each phase difference rather than cancelling it.
    conjugated = float(np.mean(np.cos(2 * phases)))

    assert cosine(mem, phasors) == pytest.approx([1.0, -1.0, conjugated])


def test_cosine_agrees_with_hrr_similarity():
    """HRR similarity is signed, so `cosine` must be too."""
    np.random.seed(0)
    mem = np.stack([HRR.new(DIM).data for _ in range(5)] + [np.zeros(DIM)])
    query = HRR.new(DIM).data

    expected = [HRR.similarity(query, row) for row in mem]

    assert cosine(mem, query) == pytest.approx(expected)


def test_pairwise_lifts_a_pairwise_similarity():
    np.random.seed(0)
    mem = np.stack([HRR.new(DIM).data for _ in range(5)])
    query = HRR.new(DIM).data

    assert pairwise(HRR.similarity)(mem, query) == pytest.approx(cosine(mem, query))


# ================= Cleanup construction =================


@pytest.mark.parametrize(("dim", "slots"), [(0, 4), (-1, 4), (8, 0), (8, -1)])
def test_cleanup_rejects_non_positive_sizes(dim, slots):
    with pytest.raises(ValueError):
        Cleanup(dim=dim, slots=slots)


def test_cleanup_starts_empty():
    assert len(Cleanup(dim=DIM, slots=4)) == 0


def test_cleanup_repr_omits_the_memory_matrix():
    assert "_mem" not in repr(Cleanup(dim=DIM, slots=4))


# ================= memorize =================


def test_memorize_returns_the_vector_it_was_given():
    memory = Cleanup(dim=DIM, slots=4)
    x = unit(0)

    assert memory.memorize(x) is x
    assert len(memory) == 1


@pytest.mark.parametrize("shape", [(1,), (DIM + 1,), (DIM, 1), (2, DIM)])
def test_memorize_rejects_the_wrong_shape(shape):
    """A `(1,)` vector used to be broadcast across the whole row."""
    memory = Cleanup(dim=DIM, slots=4)

    with pytest.raises(ValueError):
        memory.memorize(np.ones(shape))


def test_memorize_grows_past_the_pre_allocated_slots():
    memory = Cleanup(dim=DIM, slots=1)
    stored = [unit(i) for i in range(5)]

    for x in stored:
        _ = memory.memorize(x)

    assert len(memory) == 5
    assert all(memory.recall(x) == pytest.approx(x) for x in stored)


# ================= recall =================


def test_recall_of_an_empty_memory_is_none():
    assert Cleanup(dim=DIM, slots=4).recall(unit(0)) is None


def test_recall_ignores_the_unfilled_slots():
    """Empty rows score 0.0, which beat any negative activation."""
    memory = Cleanup(dim=DIM, slots=4)
    _ = memory.memorize(-unit(0))

    assert memory.recall(unit(0)) == pytest.approx(-unit(0))


def test_recall_is_not_biased_towards_larger_vectors():
    memory = Cleanup(dim=2, slots=2)
    _ = memory.memorize(vec(1.0, 0.0))
    _ = memory.memorize(vec(3.0, 3.0))

    assert memory.recall(vec(1.0, 0.0)) == pytest.approx(vec(1.0, 0.0))


def test_recall_picks_the_most_similar_vector():
    memory = Cleanup(dim=DIM, slots=4)
    for i in range(3):
        _ = memory.memorize(unit(i))

    assert memory.recall(0.9 * unit(2) + 0.1 * unit(0)) == pytest.approx(unit(2))


def test_recall_returns_a_copy():
    memory = Cleanup(dim=DIM, slots=4)
    _ = memory.memorize(unit(0))

    recalled = memory.recall(unit(0))
    recalled[:] = 99.0

    assert memory.recall(unit(0)) == pytest.approx(unit(0))


def test_recall_uses_the_given_similarity():
    memory = Cleanup(dim=DIM, slots=4, sim_function=negated_cosine)
    _ = memory.memorize(unit(0))
    _ = memory.memorize(unit(1))

    assert memory.recall(unit(0)) == pytest.approx(unit(1))


def test_cleanup_stores_complex_vectors():
    memory = Cleanup(dim=4, slots=2, dtype=np.complex128)
    phasors = np.exp(1j * np.array([0.0, 1.0, 2.0, 3.0]))
    _ = memory.memorize(phasors)

    assert memory.recall(phasors) == pytest.approx(phasors)


# ================= nearest, get and replace =================


def test_nearest_returns_the_index_and_its_activation():
    memory = Cleanup(dim=DIM, slots=4)
    for i in range(3):
        _ = memory.memorize(unit(i))

    index, activation = memory.nearest(unit(1))

    assert index == 1
    assert activation == pytest.approx(1.0)


def test_nearest_of_an_empty_memory_is_none():
    assert Cleanup(dim=DIM, slots=4).nearest(unit(0)) is None


def test_get_returns_a_copy_of_the_stored_row():
    memory = Cleanup(dim=DIM, slots=4)
    _ = memory.memorize(unit(0))

    got = memory.get(0)
    got[:] = 99.0

    assert memory.get(0) == pytest.approx(unit(0))


@pytest.mark.parametrize("index", [-1, 0, 1])
def test_get_rejects_rows_that_hold_nothing(index):
    """Slot 0 is allocated but unfilled, so reading it is still an error."""
    memory = Cleanup(dim=DIM, slots=4)

    with pytest.raises(IndexError):
        _ = memory.get(index)


def test_replace_overwrites_a_stored_vector():
    memory = Cleanup(dim=DIM, slots=4)
    _ = memory.memorize(unit(0))
    memory.replace(0, unit(1))

    assert len(memory) == 1
    assert memory.get(0) == pytest.approx(unit(1))
    assert memory.recall(unit(1)) == pytest.approx(unit(1))


def test_replace_validates_its_index_and_shape():
    memory = Cleanup(dim=DIM, slots=4)
    _ = memory.memorize(unit(0))

    with pytest.raises(IndexError):
        memory.replace(1, unit(0))

    with pytest.raises(ValueError):
        memory.replace(0, np.ones(DIM + 1))


# ================= VSACleanup =================


def test_vsa_cleanup_memorize_returns_the_same_vector_symbol():
    memory = VSACleanup(HRR, DIM)
    x = HRR(unit(0))

    assert memory.memorize(x) is x
    assert len(memory) == 1


def test_vsa_cleanup_recall_returns_a_vector_symbol():
    memory = VSACleanup(HRR, DIM)
    _ = memory.memorize(HRR(unit(0)))

    recalled = memory.recall(HRR(unit(0)))

    assert isinstance(recalled, HRR)
    assert recalled.data == pytest.approx(unit(0))


def test_vsa_cleanup_recall_of_an_empty_memory_is_none():
    assert VSACleanup(HRR, DIM).recall(HRR(unit(0))) is None


def test_vsa_cleanup_recall_does_not_alias_the_stored_vector():
    memory = VSACleanup(HRR, DIM)
    _ = memory.memorize(HRR(unit(0)))

    recalled = memory.recall(HRR(unit(0)))
    recalled.data[:] = 99.0

    assert memory.recall(HRR(unit(0))).data == pytest.approx(unit(0))


def test_vsa_cleanup_recalls_the_right_vector_symbol_from_a_noisy_query():
    np.random.seed(0)
    memory = VSACleanup(HRR, 256, slots=8)
    stored = [memory.memorize(HRR.new(256)) for _ in range(50)]

    for x in stored:
        noisy = HRR(x.data + np.random.normal(scale=0.02, size=256))
        assert memory.recall(noisy).data == pytest.approx(x.data)
