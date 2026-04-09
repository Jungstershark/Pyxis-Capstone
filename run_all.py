"""
Master runner: executes all phases sequentially with recovery.

Phase 1: Train 3 models on Singapore River gold
Phase 2: Train 3 models on internet_scraped gold (v2, 100 epochs)
Phase 3: Merge datasets, train 3 models on combined gold
Final:   Evaluate all models on gold test sets

Recovery: if a phase fails, the script logs the error and continues
to the next phase. Full stdout/stderr captured in run_all.log.

Run from Pyxis-Capstone root:
    python run_all.py
"""

import subprocess
import sys
import traceback
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / "venv310" / "Scripts" / "python.exe")
LOG_FILE = ROOT / "run_all.log"

# Also log to file
_log_fh = open(LOG_FILE, "a", encoding="utf-8")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_fh.write(line + "\n")
    _log_fh.flush()


def run_step(description, cwd, script, critical=False):
    """Run a script. Streams output to console AND log file. Returns True on success."""
    cmd = [PYTHON, script]

    log("")
    log("#" * 70)
    log(f"# {description}")
    log(f"# cwd: {cwd}")
    log(f"# cmd: {' '.join(cmd)}")
    log("#" * 70)

    try:
        # Stream output line-by-line to both console and log
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            _log_fh.write(line + "\n")
            _log_fh.flush()

        process.wait(timeout=14400)  # 4hr timeout

        if process.returncode != 0:
            log(f"")
            log(f">>> FAILED: {description} (exit code {process.returncode})")
            log(f">>> Script: {cwd / script}")
            log(f">>> You can re-run this step manually:")
            log(f">>>   cd \"{cwd}\"")
            log(f">>>   \"{PYTHON}\" {script}")
            log(f"")
            if critical:
                log(">>> Critical step failed — stopping pipeline.")
                sys.exit(process.returncode)
            return False

        log(f"OK: {description}")
        return True

    except subprocess.TimeoutExpired:
        process.kill()
        log(f">>> TIMEOUT: {description} (exceeded 4 hours)")
        log(f">>> Re-run: cd \"{cwd}\" && \"{PYTHON}\" {script}")
        return False
    except Exception as e:
        log(f">>> EXCEPTION: {description}: {e}")
        log(traceback.format_exc())
        return False


if __name__ == "__main__":
    log("=" * 70)
    log(f"STARTING run_all.py — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file: {LOG_FILE}")
    log("=" * 70)

    results = {}

    # ========================
    # Phase 1: Singapore River
    # ========================
    results["sg_train"] = run_step(
        "Phase 1: Train Singapore River (y11s@960, y11m@960, y11s@1280, 100ep each)",
        ROOT / "datasets" / "singapore_river",
        "5_train.py",
    )

    results["sg_eval"] = run_step(
        "Phase 1: Evaluate Singapore River on gold test",
        ROOT / "datasets" / "singapore_river",
        "6_evaluate.py",
    )

    # ========================
    # Phase 2: Internet Scraped v2
    # ========================
    results["is_train"] = run_step(
        "Phase 2: Retrain internet_scraped (y11s@960, y11m@960, y11s@1280, 100ep each)",
        ROOT / "datasets" / "internet_scraped",
        "5_train_v2.py",
    )

    results["is_eval"] = run_step(
        "Phase 2: Evaluate internet_scraped v2 on gold test",
        ROOT / "datasets" / "internet_scraped",
        "6_evaluate_v2.py",
    )

    # ========================
    # Phase 3: Combined dataset
    # ========================
    results["merge"] = run_step(
        "Phase 3: Merge gold datasets",
        ROOT / "datasets" / "combined",
        "1_merge_gold.py",
        critical=True,  # combined training depends on this
    )

    results["combined_train"] = run_step(
        "Phase 3: Train Combined (y11s@960, y11m@960, y11s@1280, 100ep each)",
        ROOT / "datasets" / "combined",
        "5_train.py",
    )

    results["combined_eval"] = run_step(
        "Phase 3: Evaluate Combined on gold test",
        ROOT / "datasets" / "combined",
        "6_evaluate.py",
    )

    # ========================
    # Summary
    # ========================
    log("")
    log("=" * 70)
    log("ALL PHASES COMPLETE — SUMMARY")
    log("=" * 70)

    all_ok = True
    for step, ok in results.items():
        status = "OK" if ok else "FAILED <<<"
        log(f"  {step}: {status}")
        if not ok:
            all_ok = False

    log("")
    if all_ok:
        log("All steps succeeded.")
    else:
        log("Some steps failed. Check log above for >>> FAILED lines.")
        log("Each failure includes the exact command to re-run that step.")

    log("")
    log("Results files:")
    log(f"  SG val:       datasets/singapore_river/experiments.csv")
    log(f"  SG test:      datasets/singapore_river/6_test_benchmark.csv")
    log(f"  IS v2 val:    datasets/internet_scraped/experiments_v2.csv")
    log(f"  IS v2 test:   datasets/internet_scraped/6_test_benchmark_v2.csv")
    log(f"  Combined val: datasets/combined/experiments.csv")
    log(f"  Combined test:datasets/combined/6_test_benchmark.csv")
    log("=" * 70)

    _log_fh.close()
