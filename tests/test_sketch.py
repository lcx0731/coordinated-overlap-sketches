from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import json
import math
import random

import pytest

from overlap_sketch import Config, Sketch, compare, fingerprint
from overlap_sketch.reference import full_set_reference, ideal_reference


@pytest.mark.parametrize("seed", range(16))
def test_streaming_and_partition_merge_match_full_set_sort(seed):
    rng = random.Random(seed)
    k = 3 + seed*2
    config = Config(k, seed, 8 if seed % 4 == 0 else 128)
    left = [f"id-{rng.randrange(200)}" for _ in range(300)]
    right = [f"id-{rng.randrange(300)}" for _ in range(300)]
    a, b = Sketch(config).update(iter(left)), Sketch(config).update(iter(right))
    oracle = full_set_reference(left, right, config)
    merged = a.merge(b)
    assert merged.retained == oracle.pop("retained")
    assert asdict(compare(a, b)) == oracle
    direct = Sketch(config).update(left + right)
    assert merged.dumps() == direct.dumps()
    assert b.merge(a).dumps() == merged.dumps()
    assert merged.merge(a).dumps() == merged.dumps()
    empty = Sketch(config)
    c = Sketch(config).update(left[::5])
    assert a.merge(b).merge(c).dumps() == a.merge(b.merge(c)).dumps()
    assert merged.merge(empty).dumps() == merged.dumps()
    assert len(merged.retained) <= k


@pytest.mark.parametrize("n", [0, 1, 3, 4, 20])
def test_exactness_boundary_and_duplicate_replay(n):
    config = Config(3)
    items = [str(i) for i in range(n)]
    sketch = Sketch(config).update(items)
    before = sketch.dumps()
    sketch.update(items[::-1]*3)
    assert sketch.dumps() == before
    assert sketch.exact_hash_domain == (n <= 3)
    if n <= 3:
        assert sketch.estimate() == n
        assert sketch.sample()[1] == 1
    else:
        assert len(sketch.sample()[0]) == 2
        assert sketch.retained[-1] not in sketch.sample()[0]


def test_exact_set_operations_and_empty_convention():
    config = Config(10)
    a, b = Sketch(config).update(["x", "y"]), Sketch(config).update(["y", "z"])
    result = compare(a, b)
    assert (result.union_estimate, result.intersection_estimate, result.jaccard_estimate) == (3, 1, 1/3)
    assert result.exact_hash_domain
    assert compare(Sketch(), Sketch()).jaccard_estimate == 1
    assert compare(a, Sketch(config)).intersection_estimate == 0


def test_unequal_input_sizes_use_union_threshold():
    config = Config(8, 4)
    left, right = [str(x) for x in range(300)], [str(x) for x in range(3)]
    a, b = Sketch(config).update(left), Sketch(config).update(right)
    oracle = full_set_reference(left, right, config)
    oracle.pop("retained")
    assert asdict(compare(a, b)) == oracle


def test_nonzero_intersection_can_have_zero_sample_hits():
    # Choose a high-ranked subset deterministically to expose the possibility.
    config = Config(8, 0)
    universe = [str(i) for i in range(200)]
    high = sorted(universe, key=lambda x: fingerprint(x, config))[-2:]
    a, b = Sketch(config).update(universe), Sketch(config).update(high)
    result = compare(a, b)
    assert set(universe) & set(high)
    assert result.intersection_hits == result.intersection_estimate == 0
    assert not result.exact_hash_domain


def test_toy_hash_collision_creates_false_item_intersection():
    config = Config(512, bits=8)
    seen = {}
    for i in range(257):
        identifier = str(i)
        rank = fingerprint(identifier, config)
        if rank in seen:
            left, right = seen[rank], identifier
            break
        seen[rank] = identifier
    assert left != right
    result = compare(Sketch(config).update([left]), Sketch(config).update([right]))
    assert result.exact_hash_domain and result.intersection_estimate == 1


@pytest.mark.parametrize("identifier,expected", [
    ("", "587ac28625c65aff7506f1266149e1ce"),
    ("user-17", "9a400507bae9cfead33339733c93c3e7"),
    ("水", "2441c9fbc6c959f9bfa6d5783f3ba18c"),
])
def test_hash_protocol_vectors(identifier, expected):
    assert f"{fingerprint(identifier, Config(32,7)):032x}" == expected


def test_no_unicode_normalization_and_bits_are_high_prefix():
    assert fingerprint("e\u0301", Config()) != fingerprint("é", Config())
    assert fingerprint("x", Config(bits=8)) == fingerprint("x", Config()) >> 120


@pytest.mark.parametrize("n,m,k", [(7, 3, 4), (8, 1, 3), (6, 6, 5), (6, 0, 3)])
def test_ideal_moments_from_exhaustive_label_subsets_and_beta_integrals(n,m,k):
    # Enumerate rank-label subsets. Independently integrate inverse moments of
    # the kth order statistic via beta factorials, keeping rational arithmetic.
    r = k-1
    counts = [sum(i < m for i in subset) for subset in combinations(range(n), r)]
    e_c = Fraction(sum(counts), len(counts))
    e_c2 = Fraction(sum(c*c for c in counts), len(counts))
    def beta(a,b):
        return Fraction(math.factorial(a-1)*math.factorial(b-1), math.factorial(a+b-1))
    inverse_first = beta(k-1,n-k+1)/beta(k,n-k+1)
    inverse_second = beta(k-2,n-k+1)/beta(k,n-k+1)
    mean = e_c * inverse_first
    variance = e_c2 * inverse_second - mean**2
    expected = ideal_reference(n,m,k)
    assert mean == m
    assert expected["variance"] == float(variance)
    assert expected["zero_hit_probability"] == counts.count(0)/len(counts)


@pytest.mark.parametrize("config", [Config(4,0), Config(3,1), Config(3,0,64)])
def test_incompatible_configurations_rejected(config):
    with pytest.raises(ValueError, match="identical"):
        Sketch(Config(3,0)).merge(Sketch(config))


def test_serialization_roundtrip_and_resume():
    config = Config(11,5)
    items = [str(i) for i in range(100)]
    first = Sketch(config).update(items[:70])
    restored = Sketch.loads(first.dumps())
    assert restored.dumps() == first.dumps()
    restored.update(items[70:])
    assert restored.dumps() == Sketch(config).update(items).dumps()


@pytest.mark.parametrize("change", [
    {"format":"future"}, {"extra":1}, {"k":True}, {"seed":-1}, {"bits":7},
    {"overflow":1}, {"overflow":True}, {"hashes":["0"]},
    {"hashes":["0"*32,"0"*32]}, {"bits":8,"hashes":["f"*32]},
    {"hashes":["0"*31+"2","0"*31+"1"]}, {"hashes":[None]},
])
def test_malformed_persisted_sketches_rejected(change):
    data = Sketch(Config(3)).to_dict()
    data.update(change)
    with pytest.raises(ValueError):
        Sketch.loads(json.dumps(data))


def test_duplicate_json_fields_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Sketch.loads('{"k":3,"k":4}')


@pytest.mark.parametrize("kwargs", [{"k":2},{"k":3.0},{"seed":2**64},{"seed":True},{"bits":129}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_identifier_contract():
    with pytest.raises(ValueError):
        Sketch().add(1)
    with pytest.raises(ValueError):
        Sketch().update("abc")
