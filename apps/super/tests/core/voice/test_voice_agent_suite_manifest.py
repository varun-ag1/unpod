from pathlib import Path


def test_run_tests_script_exists() -> None:
    script = Path("scripts/run_tests.sh")
    assert script.exists()
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
