import math

import numpy as np
import pytest

from plate.memory.associative import Associative, VSAAssociative, noise_threshold
from plate.memory.cleanup import cosine
from plate.vsa import HRR

DIM = 8


def unit(index: int, dim: int = DIM) -> np.ndarray:
    """The `index`th standard basis vector."""
    vector = np.zeros(dim)
    vector[index] = 1.0
    return vector


def negated_cosine(mem: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A similarity that ranks the least similar vector highest."""
    return -cosine(mem, x)


# ================= noise_threshold =================


@pytest.mark.parametrize("dim", [64, 512, 1024])
def test_noise_threshold_is_four_deviations_above_chance(dim):
    """Unrelated vectors score around 0.0 with a deviation of 1 / sqrt(dim)."""
    assert noise_threshold(dim) == pytest.approx(4.0 / math.sqrt(dim))


def test_noise_threshold_takes_a_number_of_deviations():
    assert noise_threshold(1024, z=2.0) == pytest.approx(2.0 / 32.0)


def test_noise_threshold_never_exceeds_one():
    assert noise_threshold(4) == 1.0


# ================= Associative =================


def test_theta_defaults_to_the_noise_threshold():
    assert Associative(dim=512, slots=4).theta == pytest.approx(noise_threshold(512))


def test_an_explicit_theta_is_kept():
    assert Associative(dim=512, slots=4, theta=0.5).theta == 0.5


def test_associate_then_deref_returns_the_value():
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0), unit(1))

    assert len(memory) == 1
    assert memory.deref(unit(0)) == pytest.approx(unit(1))


def test_deref_returns_the_value_rather_than_the_key():
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0), unit(1))

    assert memory.deref(unit(0)) != pytest.approx(unit(0))


def test_one_value_can_be_reached_from_several_keys():
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0), unit(2))
    memory.associate(unit(1), unit(2))

    assert len(memory) == 2
    assert memory.deref(unit(0)) == pytest.approx(unit(2))
    assert memory.deref(unit(1)) == pytest.approx(unit(2))


def test_deref_of_an_empty_memory_is_none():
    assert Associative(dim=DIM, slots=4).deref(unit(0)) is None


def test_deref_of_an_unrelated_key_is_none():
    memory = Associative(dim=DIM, slots=4, theta=0.2)
    memory.associate(unit(0), unit(1))

    assert memory.deref(unit(2)) is None


def test_a_key_at_exactly_theta_is_rejected():
    """The threshold is strict, so scoring `theta` is not similar enough."""
    query = (unit(0) + unit(1)) / math.sqrt(2.0)
    achieved = float(cosine(unit(0)[np.newaxis, :], query)[0])
    memory = Associative(dim=DIM, slots=4, theta=achieved)
    memory.associate(unit(0), unit(2))

    assert memory.deref(query) is None


def test_deref_recovers_the_value_from_a_noisy_key():
    np.random.seed(0)
    memory = Associative(dim=256, slots=4)
    key, value = HRR.new(256).data, HRR.new(256).data
    memory.associate(key, value)

    noisy = key + np.random.normal(scale=0.02, size=256)

    assert memory.deref(noisy) == pytest.approx(value)


def test_re_associating_a_key_replaces_its_value():
    """A stale duplicate would stay reachable through the similarity scan."""
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0), unit(1))
    memory.associate(unit(0), unit(2))

    assert len(memory) == 1
    assert memory.deref(unit(0)) == pytest.approx(unit(2))


def test_deref_returns_a_copy():
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0), unit(1))

    value = memory.deref(unit(0))
    value[:] = 99.0

    assert memory.deref(unit(0)) == pytest.approx(unit(1))


def test_associate_grows_past_the_pre_allocated_slots():
    memory = Associative(dim=32, slots=1)
    pairs = [(unit(i, 32), unit(i + 1, 32)) for i in range(0, 20, 2)]

    for key, value in pairs:
        memory.associate(key, value)

    assert len(memory) == len(pairs)
    assert all(memory.deref(key) == pytest.approx(value) for key, value in pairs)


def test_a_key_of_another_dtype_still_matches_exactly():
    memory = Associative(dim=DIM, slots=4)
    memory.associate(unit(0).astype(np.float32), unit(1))

    assert memory.deref(unit(0)) == pytest.approx(unit(1))


def test_an_exact_key_is_preferred_over_the_similarity_scan():
    """Even a similarity that ranks the right key last finds it exactly."""
    memory = Associative(dim=DIM, slots=4, sim_function=negated_cosine)
    memory.associate(unit(0), unit(1))
    memory.associate(unit(2), unit(3))

    assert memory.deref(unit(0)) == pytest.approx(unit(1))


def test_collides_reports_keys_close_to_a_stored_one():
    memory = Associative(dim=DIM, slots=4, theta=0.2)
    memory.associate(unit(0), unit(1))

    assert memory.collides(unit(0))
    assert memory.collides(0.9 * unit(0) + 0.1 * unit(1))
    assert not memory.collides(unit(2))


def test_collides_on_an_empty_memory_is_false():
    assert not Associative(dim=DIM, slots=4).collides(unit(0))


# ================= VSAAssociative =================


def test_alloc_returns_a_pointer_that_derefs_to_the_trace():
    np.random.seed(0)
    memory = VSAAssociative(HRR, 256)
    trace = HRR.new(256)

    ptr = memory.alloc(trace)

    assert isinstance(ptr, HRR)
    assert len(memory) == 1
    assert memory.deref(ptr).data == pytest.approx(trace.data)


def test_deref_matches_a_distinct_object_holding_equal_data():
    """`HRR` hashes by data but compares by identity, so a dict would miss."""
    np.random.seed(0)
    memory = VSAAssociative(HRR, 256)
    trace = HRR.new(256)
    ptr = memory.alloc(trace)

    assert memory.deref(HRR(ptr.data.copy())).data == pytest.approx(trace.data)


def test_associate_maps_a_given_pointer_to_a_trace():
    memory = VSAAssociative(HRR, DIM)
    memory.associate(HRR(unit(0)), HRR(unit(1)))

    assert memory.deref(HRR(unit(0))).data == pytest.approx(unit(1))


def test_deref_of_an_unrelated_pointer_is_none():
    np.random.seed(0)
    memory = VSAAssociative(HRR, 256)
    _ = memory.alloc(HRR.new(256))

    assert memory.deref(HRR.new(256)) is None


def test_vsa_deref_of_an_empty_memory_is_none():
    np.random.seed(0)

    assert VSAAssociative(HRR, 256).deref(HRR.new(256)) is None


def test_every_allocated_pointer_derefs_to_its_own_trace():
    np.random.seed(0)
    memory = VSAAssociative(HRR, 256, slots=4)
    allocated = [(memory.alloc(trace := HRR.new(256)), trace) for _ in range(20)]

    assert len(memory) == 20
    assert all(memory.deref(ptr).data == pytest.approx(t.data) for ptr, t in allocated)


def test_alloc_gives_up_when_pointers_keep_colliding():
    """Two dimensions leave too little room for distinct pointers."""
    np.random.seed(0)
    memory = VSAAssociative(HRR, 2, theta=0.3)

    with pytest.raises(RuntimeError, match="dim=2 is too small"):
        for _ in range(50):
            _ = memory.alloc(HRR.new(2))
