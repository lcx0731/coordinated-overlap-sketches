"""Build/merge persisted sketches or compare them without retaining raw identifiers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .core import Config, Sketch, compare


def _same_file(left, right):
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="one UTF-8 identifier per line; empty lines are identifiers")
    build.add_argument("input", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--k", type=int, default=256)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--bits", type=int, default=128, help="128 recommended; low values demonstrate collisions")
    merge = commands.add_parser("merge")
    merge.add_argument("inputs", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    query = commands.add_parser("compare")
    query.add_argument("left", type=Path)
    query.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            if _same_file(args.input, args.output):
                raise ValueError("input and output must be different files")
            sketch = Sketch(Config(args.k, args.seed, args.bits))
            with args.input.open(encoding="utf-8") as stream:
                for line in stream:
                    sketch.add(line.removesuffix("\n"))
            args.output.write_text(sketch.dumps(), encoding="utf-8")
        elif args.command == "merge":
            if any(_same_file(args.output, p) for p in args.inputs):
                raise ValueError("output must differ from all input sketches")
            sketch = Sketch.loads(args.inputs[0].read_text(encoding="utf-8"))
            for path in args.inputs[1:]:
                sketch = sketch.merge(Sketch.loads(path.read_text(encoding="utf-8")))
            args.output.write_text(sketch.dumps(), encoding="utf-8")
        else:
            result = compare(Sketch.loads(args.left.read_text(encoding="utf-8")),
                             Sketch.loads(args.right.read_text(encoding="utf-8")))
            print(json.dumps(asdict(result), indent=2, allow_nan=False))
    except (ValueError, TypeError, OSError, OverflowError) as error:
        print(f"overlap-sketch: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
