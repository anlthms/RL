import asyncio
import json
from pathlib import Path

import pytest

from tools.arc_augment import ArcView, sample_views
from tools.arc_sampling_harness import (
    ManifestRow,
    build_manifest,
    load_arc_rows,
    load_completed,
    load_source_ids,
    pass_at_k,
    record_key,
    render_prompt,
    run_sampling,
    score_candidate,
    summarize,
    wilson_interval,
)

TEMPLATE = "PROMPT\n{}\nANSWER NOW"


def _row(task_id="t1", test_index=0, **overrides) -> ManifestRow:
    fields = {
        "task_id": task_id,
        "test_index": test_index,
        "train_pairs": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test_input": [[3, 4]],
        "target": [[4, 3]],
        "seen_in_sources": (),
    }
    fields.update(overrides)
    return ManifestRow(**fields)


def _write_arc_fixture(tmp_path: Path, split="evaluation") -> str:
    challenges = {
        "task_b": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}, {"input": [[4]]}],
        },
        "task_a": {
            "train": [{"input": [[5, 6]], "output": [[6, 5]]}],
            "test": [{"input": [[7, 8]]}],
        },
    }
    solutions = {"task_b": [[[4]], [[5]]], "task_a": [[[8, 7]]]}
    (tmp_path / f"arc-agi_{split}_challenges.json").write_text(json.dumps(challenges))
    (tmp_path / f"arc-agi_{split}_solutions.json").write_text(json.dumps(solutions))
    return str(tmp_path)


class ScriptedClient:
    """Returns canned responses keyed by exact prompt suffix order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def complete(self, messages):
        self.prompts.append(messages[0]["content"])
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Manifest


def test_manifest_keeps_all_test_rows_of_sampled_tasks(tmp_path) -> None:
    rows = load_arc_rows(_write_arc_fixture(tmp_path), "evaluation")
    assert [(r["task_id"], r["test_index"]) for r in rows] == [
        ("task_a", 0),
        ("task_b", 0),
        ("task_b", 1),
    ]
    manifest_rows, manifest = build_manifest(rows, num_tasks=1, seed=7, source_ids={})
    # Sampling is by task id; every test row of the chosen task is present.
    chosen = {row.task_id for row in manifest_rows}
    assert len(chosen) == 1
    task = chosen.pop()
    assert [row.test_index for row in manifest_rows] == list(range(len(manifest_rows)))
    assert manifest["task_ids"] == [task]
    assert manifest["row_count"] == len(manifest_rows)

    again_rows, again = build_manifest(rows, num_tasks=1, seed=7, source_ids={})
    assert again["task_ids"] == manifest["task_ids"]


def test_manifest_labels_source_contamination_instead_of_dropping(tmp_path) -> None:
    rows = load_arc_rows(_write_arc_fixture(tmp_path), "evaluation")
    manifest_rows, manifest = build_manifest(
        rows,
        num_tasks=None,
        seed=1,
        source_ids={"sft_v4": {"task_b"}, "rl_v3": {"nothing"}},
    )
    assert manifest["seen_task_ids"] == ["task_b"]
    assert manifest["source_overlap"] == {"sft_v4": ["task_b"], "rl_v3": []}
    seen = {row.task_id: row.seen_in_sources for row in manifest_rows}
    assert seen["task_b"] == ("sft_v4",)
    assert seen["task_a"] == ()


def test_source_ids_accept_challenges_dicts_and_id_lists(tmp_path) -> None:
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"id1": {}, "id2": {}}))
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps(["id3"]))
    assert load_source_ids(f"a={dict_path}") == ("a", {"id1", "id2"})
    assert load_source_ids(f"b={list_path}") == ("b", {"id3"})
    with pytest.raises(ValueError, match="label=path"):
        load_source_ids("nolabel")


# ---------------------------------------------------------------------------
# Rendering and scoring


def test_identity_prompt_matches_the_validation_row_rendering() -> None:
    row = _row()
    prompt = render_prompt(TEMPLATE, row, ArcView())
    assert "1 2" in prompt and "3 4" in prompt
    assert prompt.startswith("PROMPT\n")
    # The canonical target never enters the prompt.
    assert "4 3" not in prompt


def test_view_prompt_and_scoring_round_trip_through_the_canonical_frame() -> None:
    row = _row()
    view = ArcView(transform="rot90", color_map=(0, 9, 8, 7, 6, 5, 4, 3, 2, 1))
    prompt = render_prompt(TEMPLATE, row, view)
    # The view frame shows transformed grids, not canonical ones.
    view_test_input = view.apply_grid(row.test_input)
    assert " ".join(str(c) for c in view_test_input[0]) in prompt
    # A model answering correctly IN THE VIEW FRAME scores an exact match.
    view_target = view.apply_grid(row.target)
    answer = "\n".join(" ".join(str(c) for c in r) for r in view_target)
    record = score_candidate(
        response=f"<answer>\n{answer}\n</answer>", row=row, view=view
    )
    assert record["grid_match"] is True
    assert record["prediction"] == row.target  # canonical frame on disk


def test_view_frame_answer_scored_raw_would_be_wrong_but_canonical_is_exact() -> None:
    # The same digits WITHOUT inversion are not the canonical target: this is
    # the never-vote-over-raw-frames property.
    row = _row(test_input=[[3, 4], [5, 6]], target=[[4, 3], [6, 5]])
    view = ArcView(transform="rot90")
    view_target = view.apply_grid(row.target)
    assert view_target != row.target
    answer = "\n".join(" ".join(str(c) for c in r) for r in view_target)
    record = score_candidate(
        response=f"<answer>\n{answer}\n</answer>", row=row, view=view
    )
    assert record["grid_match"] is True


def test_malformed_answer_scores_as_format_failure_not_a_solve() -> None:
    record = score_candidate(response="no grid here", row=_row(), view=ArcView())
    assert record["format_valid"] is False
    assert record["grid_match"] is False
    assert record["cell_match"] == 0.0
    assert record["prediction"] is None


def test_candidate_records_never_contain_target_grids() -> None:
    row = _row(target=[[9, 9, 9], [9, 1, 9]])
    record = score_candidate(
        response="<answer>\n1 2\n</answer>", row=row, view=ArcView()
    )
    serialized = json.dumps(record)
    assert "target" not in serialized
    assert json.dumps(row.target) not in serialized


# ---------------------------------------------------------------------------
# pass@k and intervals


def test_pass_at_k_matches_the_analytic_estimator() -> None:
    # n=4, c=1: pass@1 = 1/4; pass@4 = 1.
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_at_k(4, 1, 4) == pytest.approx(1.0)
    # n=10, c=2, k=5 -> 1 - C(8,5)/C(10,5) = 1 - 56/252
    assert pass_at_k(10, 2, 5) == pytest.approx(1 - 56 / 252)
    assert pass_at_k(8, 0, 8) == 0.0
    with pytest.raises(ValueError):
        pass_at_k(4, 0, 8)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = wilson_interval(3, 10)
    assert low < 0.3 < high
    assert wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Solve metrics come from grid_match only


def _record(task="t", test=0, cand=0, view="identity/c0123456789", **overrides):
    fields = {
        "task_id": task,
        "test_index": test,
        "candidate_index": cand,
        "view_id": view,
        "format_valid": True,
        "grid_match": False,
        "shape_match": True,
        "cell_match": 0.9,
        "copied_input": False,
        "prediction": [[1]],
        "seen_in_sources": [],
    }
    fields.update(overrides)
    return fields


def test_near_miss_with_high_cell_match_never_counts_toward_pass_k() -> None:
    # Every candidate is a 0.99-cell near miss: pass@k must be exactly zero.
    records = [_record(cand=i, cell_match=0.99, grid_match=False) for i in range(4)]
    report = summarize(
        records, manifest_rows=[_row(task_id="t")], pass_at=(1, 4), seed=0
    )
    view = report["views"]["identity/c0123456789"]["all_rows"]
    assert view["pass_at"]["1"]["estimate"] == 0.0
    assert view["pass_at"]["4"]["estimate"] == 0.0
    assert view["solved_rows"] == 0
    assert view["mean_cell_match"] == pytest.approx(0.99)


def test_pass_k_reports_solved_ids_and_task_level_oracle() -> None:
    records = (
        # task t1 row 0: 1 hit in 4
        [_record(task="t1", cand=i, grid_match=(i == 2)) for i in range(4)]
        # task t2 rows 0 and 1: row 0 solved, row 1 never -> task unsolved
        + [_record(task="t2", cand=i, grid_match=(i == 0)) for i in range(4)]
        + [_record(task="t2", test=1, cand=i) for i in range(4)]
    )
    manifest_rows = [
        _row(task_id="t1"),
        _row(task_id="t2"),
        _row(task_id="t2", test_index=1),
    ]
    report = summarize(records, manifest_rows=manifest_rows, pass_at=(1, 4), seed=0)
    view = report["views"]["identity/c0123456789"]["all_rows"]
    assert view["rows"] == 3
    assert view["solved_row_ids"] == ["t1:0", "t2:0"]
    assert view["solved_task_ids"] == ["t1"]
    assert view["pass_at"]["4"]["estimate"] == pytest.approx(2 / 3)
    low, high = view["pass_at"]["4"]["bootstrap_95"]
    assert 0.0 <= low <= 2 / 3 <= high <= 1.0


def test_seen_tasks_are_split_out_of_the_clean_slice() -> None:
    records = [_record(task="clean", cand=i, grid_match=True) for i in range(2)] + [
        _record(task="seen", cand=i, grid_match=True) for i in range(2)
    ]
    manifest_rows = [
        _row(task_id="clean"),
        _row(task_id="seen", seen_in_sources=("sft_v4",)),
    ]
    report = summarize(records, manifest_rows=manifest_rows, pass_at=(1,), seed=0)
    view = report["views"]["identity/c0123456789"]
    assert view["all_rows"]["rows"] == 2
    assert view["clean_rows_only"]["rows"] == 1
    assert view["clean_rows_only"]["solved_row_ids"] == ["clean:0"]


def test_voting_summary_votes_in_the_canonical_frame() -> None:
    views = sample_views(count=3, seed=2)
    view_ids = [view.view_id for view in views]
    records = []
    # Ballot 0: two views produce the canonical target, one dissents ->
    # majority vote solves the row even though one view missed.
    for i, view_id in enumerate(view_ids):
        records.append(
            _record(
                view=view_id,
                cand=0,
                grid_match=(i < 2),
                prediction=[[4, 3]] if i < 2 else [[0]],
            )
        )
    report = summarize(records, manifest_rows=[_row(task_id="t")], pass_at=(1,), seed=0)
    voting = report["voting"]
    assert voting["ballots"] == 1
    assert voting["voted"]["solved_rows"] == 1
    ordered = sorted(view_ids)
    assert set(voting["exact_correlation_between_views"]) == {
        f"{ordered[0]}|{ordered[1]}",
        f"{ordered[0]}|{ordered[2]}",
        f"{ordered[1]}|{ordered[2]}",
    }


# ---------------------------------------------------------------------------
# Sampling loop: resume + streaming records


def _harness_config(**overrides):
    from tools.arc_sampling_harness import HarnessConfig

    fields = {
        "base_url": "http://unused",
        "model": "unused",
        "data_dir": "unused",
        "split": "evaluation",
        "seed": 3,
        "num_tasks": None,
        "num_candidates": 2,
        "pass_at": (1, 2),
        "temperature": 1.0,
        "max_output_tokens": 64,
        "concurrency": 1,
        "timeout_seconds": 5.0,
        "num_views": 1,
        "fix_background": True,
        "color_permutations": True,
        "prompt_file": "unused",
        "store_responses": False,
    }
    fields.update(overrides)
    return HarnessConfig(**fields)


def test_run_sampling_streams_records_and_resumes(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(TEMPLATE)
    config = _harness_config(prompt_file=str(prompt_file))
    rows = [_row()]
    views = [ArcView()]
    candidates_path = tmp_path / "candidates.jsonl"

    client = ScriptedClient(["<answer>\n4 3\n</answer>", "junk"])
    written = asyncio.run(
        run_sampling(
            config=config,
            manifest_rows=rows,
            views=views,
            client=client,
            candidates_path=candidates_path,
        )
    )
    assert written == 2
    records = [json.loads(line) for line in candidates_path.read_text().splitlines()]
    assert {record["candidate_index"] for record in records} == {0, 1}
    assert sorted(r["grid_match"] for r in records) == [False, True]

    # Resume: nothing left to sample, existing records untouched.
    idle_client = ScriptedClient([])
    written = asyncio.run(
        run_sampling(
            config=config,
            manifest_rows=rows,
            views=views,
            client=idle_client,
            candidates_path=candidates_path,
        )
    )
    assert written == 0
    assert idle_client.prompts == []


def test_load_completed_tolerates_a_truncated_trailing_line(tmp_path) -> None:
    path = tmp_path / "candidates.jsonl"
    good = _record()
    path.write_text(json.dumps(good) + "\n" + '{"task_id": "t", "trunca')
    assert load_completed(path) == {record_key(good)}


def test_pass_at_only_reports_ks_with_enough_samples() -> None:
    records = [_record(cand=i) for i in range(2)]
    report = summarize(
        records, manifest_rows=[_row(task_id="t")], pass_at=(1, 2, 8), seed=0
    )
    view = report["views"]["identity/c0123456789"]["all_rows"]
    assert set(view["pass_at"]) == {"1", "2"}
