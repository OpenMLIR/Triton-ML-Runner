import argparse
import random
import re
import subprocess
import sys
import os
from pathlib import Path

import torch
import triton
import triton_runner

from triton_runner.compat.version import (
    is_tlx_v3_7_4,
    is_triton_geq_v3_3,
    triton_version,
    uni_triton_version,
)

DEFAULT_QUICK_DUMP_SAMPLE_SIZE = 5
DEFAULT_QUICK_DUMP_SEED = 20260417
RUNNER_PYTHON_DIR = "examples/runner/python"
QUICK_SKIP_RUNNER_CMDS = frozenset({
    f"python {RUNNER_PYTHON_DIR}/gluon/02-layouts.py",
})

def get_content(file_path):
    return Path(file_path).read_text()

def get_lines(match):
    return [line.strip() for line in match.group(1).strip().splitlines() if line.strip()]


def dedupe_keep_order(lines):
    return list(dict.fromkeys(lines))


def collect_commands(capability, quick, dump_sample_size, dump_seed):
    pattern = re.compile(rf"### sm{capability}.*?shell(.*?)```", re.DOTALL)
    if is_tlx_v3_7_4:
        runner_file_path = os.path.join("examples", "runner", "tlx", "README.md")
    else:
        runner_file_path = os.path.join("examples", "runner", f"v{uni_triton_version}", "README.md")
    match = pattern.search(get_content(runner_file_path))
    _triton_ver_tuple = tuple(int(x) for x in triton_version.split("+")[0].split("."))
    if not match or (capability == 120 and _triton_ver_tuple < (3, 3, 1)):
        return None, None

    runner_lines = get_lines(match)
    if quick:
        runner_lines = [cmd for cmd in runner_lines if cmd not in QUICK_SKIP_RUNNER_CMDS]
    runner_lines = dedupe_keep_order(runner_lines)

    generic_shell_block = re.compile(r"shell(.*?)```", re.DOTALL)
    bench_file_path = os.path.join("doc", "benchmark.md")
    bench_lines = get_lines(generic_shell_block.search(get_content(bench_file_path)))
    bench_lines = [cmd for cmd in bench_lines if "matmul/mma" not in cmd or capability >= 80]

    dump_lines = []
    if is_triton_geq_v3_3:
        debug_file_path = os.path.join("doc", "dump.md")
        for m in generic_shell_block.finditer(get_content(debug_file_path)):
            dump_lines.extend(get_lines(m))
        dump_lines = [cmd for cmd in dump_lines if "06-attention" not in cmd or capability >= 80]
        if is_tlx_v3_7_4:
            # fbtriton's make_ttgir fails to legalize python-level dump instrumentation
            # in kernels with loops; IR-level (ttir/ttgir) dumps keep working
            if any("/dump/python/" in cmd for cmd in dump_lines):
                triton_runner.color_print.yellow_print(
                    "fbtriton v3.7.4: skipping python-level dump commands (fork compiler limitation)")
            dump_lines = [cmd for cmd in dump_lines if "/dump/python/" not in cmd]
        if quick:
            rng = random.Random(dump_seed)
            dump_lines = rng.sample(dump_lines, min(dump_sample_size, len(dump_lines)))

    return runner_lines + bench_lines + dump_lines, dump_lines if quick else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a reduced regression subset.")
    parser.add_argument("--dump-sample-size", type=int, default=DEFAULT_QUICK_DUMP_SAMPLE_SIZE,
                        help="Number of dump commands to sample in --quick mode.")
    parser.add_argument("--dump-seed", type=int, default=DEFAULT_QUICK_DUMP_SEED,
                        help="Random seed for dump command sampling in --quick mode.")
    args = parser.parse_args()

    if args.dump_sample_size < 0:
        raise ValueError("--dump-sample-size must be non-negative")

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    capability = capability[0] * 10 + capability[1]
    lines, sampled_dump_lines = collect_commands(
        capability,
        quick=args.quick,
        dump_sample_size=args.dump_sample_size,
        dump_seed=args.dump_seed,
    )
    if lines is None:
        print(f"sm{capability} on triton v{triton.__version__} not found")
        return

    mode = "QUICK TEST" if args.quick else "TEST"
    triton_runner.color_print.yellow_print(f"{mode} on triton v{triton_version}")
    if args.quick and sampled_dump_lines is not None:
        print(f"quick dump sample seed={args.dump_seed} size={len(sampled_dump_lines)}")
        for cmd in sampled_dump_lines:
            print(cmd)

    fail_cmd = []
    for cmd in lines:
        triton_runner.color_print.blue_print(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("return code:", result.returncode)
        if result.returncode:
            fail_cmd.append(cmd)

    if len(fail_cmd) == 0:
        print(f"✅ ALL TEST PASS on triton v{triton_version}")
        return

    triton_runner.color_print.yellow_print(f"❌ SOME TEST FAIL on triton v{triton_version}")
    print("\n".join(fail_cmd))
    sys.exit(1)


if __name__ == "__main__":
    main()
