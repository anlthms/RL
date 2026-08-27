import json

from tools.nvarc_arc_val_materialize import main


def _write_arc_fixture(path) -> None:
    challenges = {
        "task_a": {
            "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
            "test": [{"input": [[2, 0]]}, {"input": [[0, 2]]}],
        },
        "task_b": {
            "train": [{"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]]}],
        },
    }
    solutions = {"task_a": [[[0, 2]], [[2, 0]]], "task_b": [[[6]]]}
    (path / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (path / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))


def test_main_emits_dual_rows_and_marks_contamination(tmp_path, monkeypatch) -> None:
    arc_dir = tmp_path / "arc"
    out_dir = tmp_path / "out"
    arc_dir.mkdir()
    _write_arc_fixture(arc_dir)
    template_file = tmp_path / "arc_agi.txt"
    template_file.write_text("induction task:\n{}\nreturn an <answer> block\n")
    exclude_file = tmp_path / "other_split.json"
    exclude_file.write_text(json.dumps({"task_a": {}, "task_zzz": {}}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "nvarc_arc_val_materialize.py",
            "--arc-data-path",
            str(arc_dir),
            "--output-dir",
            str(out_dir),
            "--exclude-tasks-json",
            str(exclude_file),
            "--loop-val-context-limit",
            "19456",
            "--induction-prompt-file",
            str(template_file),
        ],
    )
    main()

    rows = [
        json.loads(line) for line in (out_dir / "val.jsonl").read_text().splitlines()
    ]
    stats = json.loads((out_dir / "stats.json").read_text())

    induction = [row for row in rows if row["role"] == "induction"]
    loops = [row for row in rows if row["role"] == "induction_loop"]
    assert len(induction) == 3 and len(loops) == 3
    for row in loops:
        assert row["protocol"] == "hidden_test"
        assert row["model_context_limit"] == 19456
        assert row["train"] and len(row["test"]) == 1
    # Contamination is marked, not dropped: all rows are emitted, the clean
    # task list identifies the uncontaminated slice.
    assert stats["rows"] == 6 and stats["tasks"] == 2
    assert stats["clean_tasks"] == ["task_b"]
    assert stats["contaminated_tasks"] == 1
