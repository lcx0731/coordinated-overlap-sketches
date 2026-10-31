"""Streaming bottom-k storage. Estimators approximate an ideal continuous rank model."""

from dataclasses import dataclass
import hashlib
import heapq
import json
import re


def _integer(value, name, lower, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper or 'unbounded'}]")
    return value


@dataclass(frozen=True)
class Config:
    k: int = 256
    seed: int = 0
    bits: int = 128

    def __post_init__(self):
        _integer(self.k, "k", 3)
        _integer(self.seed, "seed", 0, 2**64 - 1)
        _integer(self.bits, "bits", 8, 128)


def fingerprint(identifier: str, config: Config) -> int:
    """UTF-8 bytes, no normalization; public seed is coordination metadata, not a secret."""
    if not isinstance(identifier, str):
        raise ValueError("identifier must be a string")
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=16,
                             key=config.seed.to_bytes(8, "big"),
                             person=b"overlap-kmv-v1").digest()
    return int.from_bytes(digest, "big") >> (128 - config.bits)


class Sketch:
    """Keeps at most k distinct integer fingerprints; duplicate ingestion is idempotent."""

    def __init__(self, config: Config = Config()):
        if not isinstance(config, Config):
            raise ValueError("config must be a Config")
        self._config = config
        self._hashes = set()
        self._heap = []  # Negated ranks implement a max-heap.
        self._overflow = False

    @property
    def config(self):
        return self._config

    @property
    def retained(self):
        return tuple(sorted(self._hashes))

    @property
    def exact_hash_domain(self):
        """True means no distinct fingerprint was discarded; it does not rule out collisions."""
        return not self._overflow

    def add(self, identifier: str):
        rank = fingerprint(identifier, self.config)
        if rank in self._hashes:
            return
        if len(self._hashes) < self.config.k:
            self._hashes.add(rank)
            heapq.heappush(self._heap, -rank)
        else:
            # A new rank proves there are > k distinct ranks, even if it is too
            # large to retain. Later repetitions of discarded ranks change nothing.
            self._overflow = True
            if rank < -self._heap[0]:
                removed = -heapq.heapreplace(self._heap, -rank)
                self._hashes.remove(removed)
                self._hashes.add(rank)

    def update(self, identifiers):
        if isinstance(identifiers, (str, bytes)):
            raise ValueError("update expects an iterable of strings, not a single string")
        for identifier in identifiers:
            self.add(identifier)
        return self

    def merge(self, other):
        """Return a fresh union sketch, leaving both sources untouched."""
        _compatible(self, other)
        ranks = self._hashes | other._hashes
        result = Sketch(self.config)
        result._overflow = self._overflow or other._overflow or len(ranks) > self.config.k
        result._hashes = set(heapq.nsmallest(self.config.k, ranks))
        result._heap = [-rank for rank in result._hashes]
        heapq.heapify(result._heap)
        return result

    def sample(self):
        """Return usable ranks and threshold; an overflowing sketch excludes its kth rank."""
        ranks = self.retained
        if self.exact_hash_domain:
            return ranks, 1.0
        # Open-interval grid avoids a zero denominator; ranking stays integer-exact.
        threshold = (ranks[-1] + 1) / ((1 << self.config.bits) + 1)
        return ranks[:-1], threshold

    def estimate(self):
        ranks, threshold = self.sample()
        return len(ranks) / threshold

    def to_dict(self):
        return {"format": "overlap-kmv-v1", "k": self.config.k, "seed": self.config.seed,
                "bits": self.config.bits, "overflow": self._overflow,
                "hashes": [f"{rank:032x}" for rank in self.retained]}

    def dumps(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def loads(cls, text):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        data = json.loads(text, object_pairs_hook=unique)
        keys = {"format", "k", "seed", "bits", "overflow", "hashes"}
        if not isinstance(data, dict) or set(data) != keys or data["format"] != "overlap-kmv-v1":
            raise ValueError("unsupported sketch format or fields")
        config = Config(data["k"], data["seed"], data["bits"])
        if type(data["overflow"]) is not bool or not isinstance(data["hashes"], list):
            raise ValueError("overflow must be boolean and hashes must be a list")
        if len(data["hashes"]) > config.k:
            raise ValueError("more retained hashes than k")
        ranks = []
        for value in data["hashes"]:
            if not isinstance(value, str) or re.fullmatch("[0-9a-f]{32}", value) is None:
                raise ValueError("hashes must be lowercase 32-digit hexadecimal strings")
            rank = int(value, 16)
            if rank >= (1 << config.bits):
                raise ValueError("hash out of range for configured bits")
            ranks.append(rank)
        if ranks != sorted(set(ranks)):
            raise ValueError("hashes must be sorted and unique")
        if data["overflow"] and (len(ranks) != config.k or config.k >= (1 << config.bits)):
            raise ValueError("inconsistent overflow state")
        result = cls(config)
        result._hashes = set(ranks)
        result._heap = [-rank for rank in ranks]
        heapq.heapify(result._heap)
        result._overflow = data["overflow"]
        return result


def _compatible(left, right):
    if not isinstance(right, Sketch) or left.config != right.config:
        raise ValueError("sketches must have identical k, seed and hash bits")


@dataclass(frozen=True)
class Comparison:
    union_estimate: float
    intersection_estimate: float
    jaccard_estimate: float
    sample_size: int
    intersection_hits: int
    threshold: float
    exact_hash_domain: bool


def compare(left: Sketch, right: Sketch) -> Comparison:
    """Estimate intersection from one coordinated UNION sample, not sample intersection alone."""
    merged = left.merge(right)
    sample, threshold = merged.sample()
    hits = sum(rank in left._hashes and rank in right._hashes for rank in sample)
    return Comparison(len(sample) / threshold, hits / threshold,
                      hits / len(sample) if sample else 1.0,
                      len(sample), hits, threshold, merged.exact_hash_domain)
