import argparse
import os
import subprocess
import sys
from pathlib import Path

# All supported versions. 3.3.x uses 3.3.1, 3.5.x uses 3.5.1.
ALL_VERSIONS = ["3.8.0", "3.7.0", "3.6.0", "3.5.1", "3.4.0", "3.3.1", "3.2.0", "3.1.0", "3.0.0"]

DEFAULT_QUICK_DUMP_SAMPLE_SIZE = 5
DEFAULT_QUICK_DUMP_SEED = 20260417


parser = argparse.ArgumentParser()
parser.add_argument("versions", nargs="*", default=ALL_VERSIONS)
parser.add_argument("--quick", action="store_true", help="Run the reduced regression subset in test/test.py.")
parser.add_argument("--dump-sample-size", type=int, default=DEFAULT_QUICK_DUMP_SAMPLE_SIZE,
                    help="Number of dump commands to sample when --quick is enabled.")
parser.add_argument("--dump-seed", type=int, default=DEFAULT_QUICK_DUMP_SEED,
                    help="Random seed for dump command sampling when --quick is enabled.")
args = parser.parse_args()

versions = args.versions

python_exe = sys.executable

passed = []
failed = []

for ver in versions:
    print(f"\n==========================================")
    print(f"Installing triton=={ver}")
    print(f"==========================================")
    subprocess.run([python_exe, "-m", "pip", "install", "-q", f"triton=={ver}"], check=True)

    print(f"Running test on triton=={ver}...")
    test_cmd = [python_exe, "test/test.py"]
    if args.quick:
        test_cmd.extend([
            "--quick",
            "--dump-sample-size",
            str(args.dump_sample_size),
            "--dump-seed",
            str(args.dump_seed),
        ])
    proc = subprocess.Popen(
        test_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    fail_cmds = []
    collecting = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "❌ SOME TEST FAIL" in line:
            collecting = True
            continue
        if collecting and line.strip():
            fail_cmds.append(line.strip())
    proc.wait()
    if proc.returncode == 0:
        passed.append(ver)
    else:
        failed.append((ver, fail_cmds))

summary_lines = []
summary_lines.append("==========================================")
summary_lines.append("REGRESSION SUMMARY")
summary_lines.append("==========================================")
if passed:
    summary_lines.append(f"✅ PASS: {' '.join(passed)}")
if failed:
    for ver, fail_cmds in failed:
        summary_lines.append(f"❌ FAIL: triton=={ver}")
        for cmd in fail_cmds:
            summary_lines.append(f"  - {cmd}")

summary = "\n".join(summary_lines)
print(f"\n{summary}")

Path("regression_result.txt").write_text(summary + "\n")

if failed:
    sys.exit(1)
