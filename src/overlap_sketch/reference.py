"""Independent full-set sorting and ideal continuous-rank benchmark formulas."""

import math

from .core import Config, fingerprint


def full_set_reference(left, right, config: Config):
    """Materializes all fingerprints: a small-data oracle, never a streaming method."""
    a, b = {fingerprint(x, config) for x in left}, {fingerprint(x, config) for x in right}
    union = sorted(a | b)
    exact = len(union) <= config.k
    sample = union if exact else union[:config.k - 1]
    threshold = 1.0 if exact else (union[config.k - 1] + 1) / (2**config.bits + 1)
    hits = len(set(sample) & a & b)
    return {"retained": tuple(union[:config.k]), "exact_hash_domain": exact,
            "union_estimate": len(sample)/threshold, "intersection_estimate": hits/threshold,
            "jaccard_estimate": hits/len(sample) if sample else 1.0,
            "sample_size": len(sample), "intersection_hits": hits, "threshold": threshold}


def ideal_reference(n: int, m: int, k: int):
    """Known-size benchmark, not a confidence interval or a guarantee for finite hashes.

    n: union cardinality, m: subset/intersection cardinality. Assumes independent
    continuous uniform ranks, no collisions, fixed sets independent of ranks.
    """
    if any(type(x) is not int for x in (n, m, k)) or not 0 <= m <= n or k < 3:
        raise ValueError("require integer n >= m >= 0 and k >= 3")
    if n <= k:
        return {"mean": float(m), "variance": 0.0, "zero_hit_probability": float(m == 0)}
    r = k - 1
    zero = math.comb(n-m, r) / math.comb(n, r) if n-m >= r else 0.0
    return {"mean": float(m), "variance": m * (n-k+1) / (k-2), "zero_hit_probability": zero}
