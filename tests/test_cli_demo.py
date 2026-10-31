import json
from pathlib import Path
import subprocess
import sys

from overlap_sketch import Config, Sketch
from overlap_sketch.cli import main
from overlap_sketch.demo import collision_audit, merge_audit, seed_sweep


def test_build_merge_compare_cli(tmp_path, capsys):
    paths = []
    for i, text in enumerate(["a\nb\n\n", "b\nc\n"]):
        source = tmp_path/f"ids{i}.txt"
        source.write_text(text)
        output = tmp_path/f"sketch{i}.json"
        assert main(["build",str(source),"--output",str(output),"--k","10"]) == 0
        paths.append(output)
    assert main(["compare",str(paths[0]),str(paths[1])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["union_estimate"] == 4
    assert result["intersection_estimate"] == 1
    output = tmp_path/"union.json"
    assert main(["merge",*(str(p) for p in paths),"--output",str(output)]) == 0
    assert Sketch.loads(output.read_text()).estimate() == 4


def test_line_endings_preserve_spaces_and_empty_identifier(tmp_path):
    source, output = tmp_path/"ids.txt", tmp_path/"sketch.json"
    source.write_bytes(b" id \r\n\r\nid\r\n")
    assert main(["build",str(source),"--output",str(output)]) == 0
    expected = Sketch().update([" id ","","id"])
    assert output.read_text() == expected.dumps()


def test_no_in_place_or_hardlink_overwrite(tmp_path, capsys):
    source, linked = tmp_path/"input.txt", tmp_path/"alias.txt"
    source.write_text("abc\n")
    linked.hardlink_to(source)
    for output in (source, linked):
        assert main(["build",str(source),"--output",str(output)]) == 2
    assert source.read_text() == "abc\n"
    assert "different files" in capsys.readouterr().err


def test_cli_configuration_mismatch_exit_code(tmp_path):
    left, right = tmp_path/"left.json", tmp_path/"right.json"
    left.write_text(Sketch(Config(seed=0)).dumps())
    right.write_text(Sketch(Config(seed=1)).dumps())
    completed = subprocess.run([sys.executable,"-m","overlap_sketch.cli","compare",str(left),str(right)],
                               capture_output=True,text=True)
    assert completed.returncode == 2
    assert "identical" in completed.stderr and "Traceback" not in completed.stderr


def test_cli_missing_file(tmp_path, capsys):
    assert main(["compare",str(tmp_path/"none"),str(tmp_path/"missing")]) == 2
    assert "overlap-sketch:" in capsys.readouterr().err


def test_demo_merge_and_collisions_are_checked_against_truth():
    merge = merge_audit()
    assert merge["direct_equals_merged"] and merge["partition_replay_idempotent"]
    assert merge["retained_fingerprints"] == 128
    collision = collision_audit()
    assert collision["results"][0]["intersection_estimate"] > 0
    assert collision["results"][1]["intersection_estimate"] == 0
    assert collision["results"][1]["union_estimate"] == 600


def test_small_seed_sweep_is_reproducible_and_records_finite_metrics():
    first = seed_sweep(repetitions=4,n=100,overlaps=(2,20),capacities=(8,16))
    second = seed_sweep(repetitions=4,n=100,overlaps=(2,20),capacities=(8,16))
    assert first == second
    assert len(first) == 4
    json.dumps(first, allow_nan=False)
