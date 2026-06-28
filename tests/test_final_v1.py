"""final_v1 單元測試（不需 GPU / API）。

  uv run python -m pytest tests/test_final_v1.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.fingerprint import (
    atomic_write_json,
    canonical_json,
    load_json,
    normalize_text,
    text_fingerprint,
)
from src.step4_cv.group_folds import (
    attach_text_fingerprint,
    audit_cross_fold,
    create_group_aware_folds,
    detect_label_conflicts,
)
from src.step5_reasoning.reason import build_xml_answer
from src.step6_cv.common import extract_label
from src.step6_cv.job_matrix import build_jobs, jobs_for_machine, make_job_id


# ---------------- fingerprint ----------------
def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a\n b\t c ") == "a b c"
    assert text_fingerprint("a  b") == text_fingerprint("a\nb")
    assert text_fingerprint("a b") != text_fingerprint("a c")


def test_canonical_json_is_sorted():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_atomic_write_roundtrip(tmp_path):
    path = tmp_path / "x" / "y.json"
    atomic_write_json(path, {"k": [1, 2, 3]})
    assert load_json(path) == {"k": [1, 2, 3]}


# ---------------- group folds ----------------
def _records():
    recs = []
    labels = (
        ["Non-ESG"] * 30
        + ["Climate Change"] * 9
        + ["Human Capital"] * 9
        + ["Community Relations"] * 6
        + ["Natural Capital"] * 6
    )
    for i, label in enumerate(labels):
        recs.append({"paragraph_id": f"P{i:04d}", "combined_text": f"text body number {i}", "label": label})
    return recs


def test_group_folds_no_leakage_and_full_coverage():
    recs = _records()
    attach_text_fingerprint(recs)
    folds = create_group_aware_folds(recs, n_splits=3, shuffle=True, random_state=42)
    assert len(folds) == 3
    audit = audit_cross_fold(folds, population_size=len(recs))
    assert audit["ok"], audit
    fps = [r["text_fingerprint"] for r in recs]
    for fold in folds:
        train_fps = {fps[i] for i in fold["train_indices"]}
        val_fps = {fps[i] for i in fold["val_indices"]}
        assert not (train_fps & val_fps)


def test_duplicate_text_group_stays_together():
    recs = _records()
    # 兩筆相同文本、相同標籤 -> 同 group，不可被拆開
    recs.append({"paragraph_id": "DUP_A", "combined_text": "shared duplicate text", "label": "Non-ESG"})
    recs.append({"paragraph_id": "DUP_B", "combined_text": "shared duplicate text", "label": "Non-ESG"})
    attach_text_fingerprint(recs)
    folds = create_group_aware_folds(recs, n_splits=3, shuffle=True, random_state=42)
    dup_fp = text_fingerprint("shared duplicate text")
    fps = [r["text_fingerprint"] for r in recs]
    for fold in folds:
        in_train = dup_fp in {fps[i] for i in fold["train_indices"]}
        in_val = dup_fp in {fps[i] for i in fold["val_indices"]}
        assert not (in_train and in_val)


def test_conflict_detection():
    recs = [
        {"paragraph_id": "A", "combined_text": "same text", "label": "Non-ESG"},
        {"paragraph_id": "B", "combined_text": "same text", "label": "Climate Change"},
        {"paragraph_id": "C", "combined_text": "other", "label": "Non-ESG"},
    ]
    attach_text_fingerprint(recs)
    conflicts = detect_label_conflicts(recs)
    assert len(conflicts) == 1
    assert set(conflicts[0]["labels"]) == {"Non-ESG", "Climate Change"}


def test_no_conflict_when_same_label():
    recs = [
        {"paragraph_id": "A", "combined_text": "same text", "label": "Non-ESG"},
        {"paragraph_id": "B", "combined_text": "same text", "label": "Non-ESG"},
    ]
    attach_text_fingerprint(recs)
    assert detect_label_conflicts(recs) == []


# ---------------- xml answer / label parse ----------------
def test_xml_answer_roundtrip():
    answer = build_xml_answer("because it discusses emissions", "Climate Change")
    assert "<reasoning>" in answer and "<label>Climate Change</label>" in answer
    assert extract_label(answer) == "Climate Change"


def test_extract_label_ignores_reasoning_mentions():
    answer = build_xml_answer("Not about Non-ESG topics; it is about board governance", "Corporate Governance")
    assert extract_label(answer) == "Corporate Governance"


# ---------------- job matrix ----------------
def _jm_cfg():
    return {
        "experiments": ["base", "balance", "combined"],
        "folds": [0, 1, 2],
        "machines": {"machine_a": {"models": ["gemma", "llama"]}, "machine_b": {"models": ["qwen", "ministral"]}},
    }


def test_job_matrix_count_and_unique():
    jobs = build_jobs("final_v1", _jm_cfg())
    assert len(jobs) == 3 * 4 * 3
    assert len({j["job_id"] for j in jobs}) == len(jobs)


def test_job_matrix_machine_split():
    jobs = build_jobs("final_v1", _jm_cfg())
    assert len(jobs_for_machine(jobs, "machine_a")) == 3 * 2 * 3
    assert len(jobs_for_machine(jobs, "machine_b")) == 3 * 2 * 3


def test_job_id_format():
    assert make_job_id("final_v1", "base", "gemma", 0) == "final_v1__base__gemma__fold0__finetune_infer"
