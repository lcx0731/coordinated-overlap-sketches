"""Deterministic synthetic identifier sets and a fixed sweep over public hash seeds."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .core import Config, Sketch, compare
from .reference import full_set_reference, ideal_reference


def sets(n, m):
    split = (n + m) // 2
    return ([f"id-{i}" for i in range(split)], [f"id-{i}" for i in range(split-m, n)])


def summary_stats(values, truth):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "relative_mean_error": float(values.mean()/truth - 1),
            "relative_rmse": float(np.sqrt(np.mean((values-truth)**2))/truth)}


def seed_sweep(repetitions=160, n=4000, overlaps=(4, 400), capacities=(32, 128, 512)):
    rows = []
    for m in overlaps:
        left, right = sets(n, m)
        assert len(set(left) | set(right)) == n and len(set(left) & set(right)) == m
        for k in capacities:
            union, intersection, jaccard, naive, zeros = [], [], [], [], []
            for seed in range(repetitions):
                config = Config(k, seed)
                a, b = Sketch(config).update(left), Sketch(config).update(right)
                result = compare(a, b)
                union.append(result.union_estimate)
                intersection.append(result.intersection_estimate)
                jaccard.append(result.jaccard_estimate)
                naive.append(len(set(a.retained) & set(b.retained))/k)
                zeros.append(result.intersection_hits == 0)
            ideal = ideal_reference(n, m, k)
            rows.append({"union_truth": n, "intersection_truth": m, "k": k,
                         "seeds": repetitions, "union": summary_stats(union, n),
                         "intersection": summary_stats(intersection, m),
                         "jaccard_mean": float(np.mean(jaccard)), "jaccard_truth": m/n,
                         "naive_retained_intersection_over_k_mean": float(np.mean(naive)),
                         "zero_hit_fraction": float(np.mean(zeros)),
                         "ideal_zero_hit_probability": ideal["zero_hit_probability"],
                         "ideal_intersection_relative_sd": math.sqrt(ideal["variance"])/m,
                         "ideal_union_relative_sd": math.sqrt(ideal_reference(n, n, k)["variance"])/n})
    return rows


def merge_audit():
    config = Config(128, 23)
    ids = [f"event-{i}" for i in range(5000)]
    stream = ids + ids[::3] + ids[::-1]
    direct = Sketch(config).update(iter(stream))
    shards = [Sketch(config).update(stream[i::8]) for i in range(8)]
    merged = Sketch(config)
    for shard in reversed(shards):
        merged = merged.merge(shard)
    assert merged.dumps() == direct.dumps()
    assert merged.merge(shards[0]).dumps() == direct.dumps()
    oracle = full_set_reference(stream, [], config)
    assert direct.retained == oracle["retained"]
    return {"input_events": len(stream), "distinct_identifiers": len(ids), "partitions": 8,
            "retained_fingerprints": len(merged.retained), "estimate": merged.estimate(),
            "serialized_json_bytes": len(merged.dumps().encode("utf-8")),
            "direct_equals_merged": True, "partition_replay_idempotent": True}


def collision_audit():
    left = [f"left-{i}" for i in range(300)]
    right = [f"right-{i}" for i in range(300)]
    rows = []
    for bits in (8, 128):
        config = Config(1024, 0, bits)
        a, b = Sketch(config).update(left), Sketch(config).update(right)
        result = compare(a, b)
        assert result.exact_hash_domain
        rows.append({"hash_bits": bits, **asdict(result)})
    return {"true_union": 600, "true_intersection": 0, "results": rows}


def render(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    rows = summary["seed_sweep"]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
        colors = {4: "#c26339", 400: "#238478"}
        for m in (4, 400):
            group = [row for row in rows if row["intersection_truth"] == m]
            k = [row["k"] for row in group]
            axes[0].plot(k, [row["intersection"]["relative_rmse"] for row in group],
                         "o-", color=colors[m], label=f"Intersection M={m}")
            axes[0].plot(k, [row["ideal_intersection_relative_sd"] for row in group],
                         "--", color=colors[m], alpha=.65)
        group = [row for row in rows if row["intersection_truth"] == 4]
        k = [row["k"] for row in group]
        axes[0].plot(k, [row["union"]["relative_rmse"] for row in group], "o-",
                     color="#2d6695", label="Union N=4,000")
        axes[0].plot(k, [row["ideal_union_relative_sd"] for row in group], "--", color="#2d6695", alpha=.65)
        axes[0].set(xscale="log", yscale="log", xlabel="Capacity k", ylabel="Relative error",
                    title="Small intersections are much noisier", xticks=k, xticklabels=k)
        axes[0].legend(fontsize=9)
        axes[0].text(.5, -.22, "Solid: observed RMSE · Dashed: ideal SD", ha="center", transform=axes[0].transAxes, fontsize=10)

        axes[1].plot(k, [row["zero_hit_fraction"] for row in group], "o-", color=colors[4], label="Observed seed fraction")
        axes[1].plot(k, [row["ideal_zero_hit_probability"] for row in group], "s--", color="#3c4752", label="Ideal sampling probability")
        axes[1].set(xscale="log", xlabel="Capacity k", ylabel="Fraction / probability of zero hits",
                    title="Zero estimate can miss a real overlap", xticks=k, xticklabels=k, ylim=(0, 1.05))
        axes[1].legend(fontsize=9, loc="lower left")
        axes[1].text(.5, -.22, "The actual intersection contains 4 identifiers.", ha="center", transform=axes[1].transAxes, fontsize=10)

        group = [row for row in rows if row["intersection_truth"] == 400]
        axes[2].plot(k, [row["jaccard_mean"] for row in group], "o-", color=colors[400], label="Coordinated union sample")
        axes[2].plot(k, [row["naive_retained_intersection_over_k_mean"] for row in group], "s-", color="#a65163", label="Naive shared hashes / k")
        axes[2].axhline(.1, ls="--", color="#3c4752", label="True Jaccard = 0.1")
        axes[2].set(xscale="log", xlabel="Capacity k", ylabel="Mean similarity estimate",
                    title="Intersecting local samples is not enough", xticks=k, xticklabels=k, ylim=(0, .23))
        axes[2].legend(fontsize=9, loc="lower left")
        for ax in axes:
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
        fig.suptitle(f"Coordinated bottom-k sketches · synthetic sets · {summary['seeds']} fixed hash seeds", fontsize=15)
        fig.savefig(destination, dpi=160, metadata={"Software": "coordinated-overlap-sketches"})
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"data": "Deterministic synthetic identifier sets; no external or private data.",
               "seeds": 160, "seed_sweep": seed_sweep(), "merge": merge_audit(),
               "collision": collision_audit()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    left, right = sets(4000, 400)
    for name, ids in (("left", left), ("right", right)):
        (args.output / f"{name}.json").write_text(Sketch(Config(128, 7)).update(ids).dumps())
    render(summary, args.output / "overlap-diagnostics.png")
    print(json.dumps({"merge": summary["merge"], "collision": summary["collision"],
                      "seed_sweep": summary["seed_sweep"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
