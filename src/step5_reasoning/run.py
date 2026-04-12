from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.step5_reasoning.common import load_step5_config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


def main() -> None:
    config = load_step5_config()
    run_reasoning = bool(config["generation"].get("run_reasoning_generation", True))
    run_sft = bool(config["sft"].get("run_sft_build", True))
    script_dir = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[2]

    if run_reasoning:
        logger.info("Running Step 5 reasoning generation")
        subprocess.run(["uv", "run", "python", str(script_dir / "generate_reasoning.py")], cwd=str(project_root), check=True)

    if run_sft:
        logger.info("Running Step 5 SFT assembly")
        subprocess.run(["uv", "run", "python", str(script_dir / "build_sft.py")], cwd=str(project_root), check=True)


if __name__ == "__main__":
    main()
