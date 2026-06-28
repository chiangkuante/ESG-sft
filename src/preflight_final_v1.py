"""final_v1 preflight：在啟動昂貴的 Step 2-5 前做純驗證，不啟動任何訓練/推論。

檢查項目（對應 .todo/todo.md 3.1）：
- 兩份 ground truth 重新計算 SHA-256，必須與 manifest 相符
- base=500、blance=480、九類標籤合法、paragraph_id 不重複、文本不為空
- Step 1 paragraphs.json 與 parsing_stats.json 一致
- API key / GPU / 磁碟空間 / uv.lock 環境存在（不把 secret 寫進任何輸出）
- config schema dry validation

只列印報告與回傳 exit code，不寫含 secret 的檔案。
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.fingerprint import (
    expected_gt_sha256,
    load_config,
    load_ground_truth_manifest,
    load_json,
    resolve_path,
    sha256_file,
)
from src.step4_cv.common import ESG_CATEGORIES, canonicalize_label

VALID = set(ESG_CATEGORIES)


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        if condition:
            print(f"  [OK]   {ok_msg}")
            return True
        print(f"  [FAIL] {fail_msg}")
        self.failures.append(fail_msg)
        return False

    def warn(self, condition: bool, ok_msg: str, warn_msg: str) -> None:
        if condition:
            print(f"  [OK]   {ok_msg}")
        else:
            print(f"  [WARN] {warn_msg}")
            self.warnings.append(warn_msg)


def validate_csv(path: Path, expected_rows: int, checker: Checker, name: str) -> None:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checker.check(len(rows) == expected_rows, f"{name} rows={len(rows)}", f"{name} rows={len(rows)} != {expected_rows}")

    pids = [r.get("paragraph_id", "") for r in rows]
    checker.check(len(pids) == len(set(pids)), f"{name} paragraph_id unique", f"{name} duplicate paragraph_id")

    bad_label = [r.get("paragraph_id") for r in rows if canonicalize_label(r.get("label")) not in VALID]
    checker.check(not bad_label, f"{name} all labels valid", f"{name} invalid labels: {bad_label[:5]}")

    empty_text = [r.get("paragraph_id") for r in rows if not (r.get("combined_text") or "").strip()]
    checker.check(not empty_text, f"{name} no empty combined_text", f"{name} empty text: {empty_text[:5]}")


def main() -> int:
    config = load_config()
    checker = Checker()

    print("== Ground truth fingerprints ==")
    gt_cfg = config["pipeline"]["ground_truth"]
    gt_manifest = load_ground_truth_manifest(config)
    base_csv = resolve_path(gt_cfg["base_csv"])
    balance_csv = resolve_path(gt_cfg["balance_csv"])
    for csv_path, dataset in ((base_csv, "base"), (balance_csv, "blance")):
        if not checker.check(csv_path.exists(), f"{dataset} csv exists", f"{dataset} csv missing: {csv_path}"):
            continue
        actual = sha256_file(csv_path)
        expected = expected_gt_sha256(gt_manifest, dataset)
        checker.check(actual == expected, f"{dataset} SHA-256 matches manifest", f"{dataset} SHA-256 mismatch")

    print("\n== Ground truth content ==")
    if base_csv.exists():
        validate_csv(base_csv, int(gt_cfg["base_rows"]), checker, "base")
    if balance_csv.exists():
        validate_csv(balance_csv, int(gt_cfg["balance_rows"]), checker, "blance")

    print("\n== Step 1 consistency ==")
    paragraphs_path = resolve_path(config["pipeline"]["step1"]["paragraphs"])
    stats_path = resolve_path(config["pipeline"]["step1"]["parsing_stats"])
    if checker.check(paragraphs_path.exists(), "paragraphs.json exists", f"missing {paragraphs_path}") and stats_path.exists():
        stats = load_json(stats_path)
        # 大檔：用 ijson 不可得時退回計數 list 長度
        paragraphs = load_json(paragraphs_path)
        n = len(paragraphs)
        expected_n = int(stats.get("total_paragraphs", -1))
        checker.check(n == expected_n, f"paragraphs={n} == parsing_stats.total_paragraphs", f"paragraphs={n} != stats {expected_n}")
        del paragraphs

    print("\n== Environment ==")
    for env_name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        present = bool((Path(".env").exists() and env_name in Path(".env").read_text()) or __import__("os").getenv(env_name))
        checker.warn(present, f"{env_name} present", f"{env_name} not set (needed for Step 5/API baseline)")

    try:
        import torch

        checker.warn(torch.cuda.is_available(), f"GPU available: {torch.cuda.get_device_name(0)}", "no CUDA GPU (Step 2/6 need GPU)")
    except Exception as exc:  # noqa: BLE001
        checker.warn(False, "", f"torch import failed: {exc}")

    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    checker.warn(free_gb > 10, f"disk free {free_gb:.1f} GB", f"low disk: {free_gb:.1f} GB")
    checker.check(resolve_path("uv.lock").exists(), "uv.lock present", "uv.lock missing")

    print("\n== Config schema ==")
    checker.check(config.get("pipeline", {}).get("run_id") == "final_v1", "pipeline.run_id=final_v1", "pipeline.run_id != final_v1")
    s5 = config.get("step5_reasoning", {})
    checker.check(bool(s5.get("sft", {}).get("run_sft_build")), "step5.sft.run_sft_build=true", "run_sft_build not true")
    abl = config.get("step6_cv", {}).get("finetune", {}).get("ablation", {})
    checker.check(abl.get("label_only") is False, "ablation.label_only=false", "label_only != false")
    checker.check(abl.get("non_esg_cap") is None, "ablation.non_esg_cap=null", "non_esg_cap != null")
    checker.check(
        config.get("step4_cv", {}).get("pseudo_labels", {}).get("enabled") is False
        and config.get("step4_cv", {}).get("synthetic_generation", {}).get("enabled") is False,
        "pseudo_labels & synthetic disabled",
        "pseudo/synthetic not disabled",
    )

    print("\n" + "=" * 50)
    if checker.failures:
        print(f"PREFLIGHT FAILED: {len(checker.failures)} error(s), {len(checker.warnings)} warning(s)")
        return 1
    print(f"PREFLIGHT PASSED ({len(checker.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
