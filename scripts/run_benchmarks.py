#!/usr/bin/env python3
"""
Automated benchmark runner for InfiniLM.
Iterates over models, starts/stops the server, runs benchmarks, and prints a summary table.
"""

import argparse
import csv
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================
# Default model names (only folder names, will be joined with --model-dir)
# ============================================================

DEFAULT_MODEL_NAMES = [
    "Baichuan2-7B-Chat",
    "chatglm3-6b",
    "DeepSeek-R1-Distill-Qwen-7B",
    "GLM-4-9B-0414",
    "internlm3-8b-instruct",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "Qwen2.5-0.5B-Instruct",
    "Qwen3-4B-Instruct-2507",
]

# ============================================================
# Data
# ============================================================

@dataclass
class BenchResult:
    model_name: str
    tp: int
    status: str = "OK"
    req_throughput: Optional[float] = None
    input_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    total_throughput: Optional[float] = None
    mean_e2e_latency: Optional[float] = None
    mean_ttft: Optional[float] = None
    p99_ttft: Optional[float] = None
    mean_itl: Optional[float] = None
    p99_itl: Optional[float] = None

# ============================================================
# Helpers
# ============================================================

METRIC_PATTERNS = {
    "req_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
    "input_throughput": r"Input token throughput \(tok/s\):\s+([\d.]+)",
    "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "total_throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "mean_e2e_latency": r"Mean E2E Latency \(ms\):\s+([\d.]+)",
    "mean_ttft": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "p99_ttft": r"P99 TTFT \(ms\):\s+([\d.]+)",
    "mean_itl": r"Mean ITL \(ms\):\s+([\d.]+)",
    "p99_itl": r"P99 ITL \(ms\):\s+([\d.]+)",
}


def build_model_path(model_dir: str, model_name: str) -> str:
    """Join model_dir and model_name, ensuring no double slashes."""
    return os.path.join(model_dir.rstrip("/"), model_name)


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def wait_for_server(host: str, port: int, timeout: int,
                    proc=None, log_path: str = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check if the server process has already exited
        if proc is not None and proc.poll() is not None:
            print(f"\n  !!! Server DIED (rc={proc.returncode})")
            if log_path and os.path.exists(log_path):
                with open(log_path) as f:
                    lines = f.readlines()
                print(f"  Last 20 lines of {log_path}:")
                for line in lines[-20:]:
                    print(f"      {line.rstrip()}")
            return False

        if is_port_open(host, port):
            # Extra grace period: port is open but handler may not be ready
            time.sleep(2)
            return True
        time.sleep(3)
    return False


def start_server(model_path: str, tp: int, device: str, server_script: str,
                 host: str, port: int, enable_paged_attn: bool = True):
    env = os.environ.copy()

    # Strip trailing slash for consistency
    model_path = model_path.rstrip("/")

    cmd = [
        sys.executable, server_script,
        "--device", device,
        f"--model={model_path}",
        f"--tp={tp}",
        "--host", host,
        "--port", str(port),
    ]
    if enable_paged_attn:
        cmd.append("--enable-paged-attn")

    name = os.path.basename(model_path)
    log_path = f"/tmp/infinilm_{name}.log"
    log_file = open(log_path, "w")

    print(f"  Server log: {log_path}")
    print(f"  Server cmd: {' '.join(cmd)}")

    # Write stdout/stderr to a log file to avoid PIPE buffer deadlock.
    # Model loading logs can easily exceed the 64KB pipe buffer and cause
    # the server process to block on write(), never completing init.
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc, log_file, log_path


def stop_server(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def run_benchmark(model_path: str, bench_script: str, host: str, port: int,
                  num_prompts: int, input_len: int, output_len: int,
                  timeout: int) -> str:
    # Strip trailing slash -- sglang.bench_serving does not accept it
    model_path = model_path.rstrip("/")

    # Support both module form ("-m sglang.bench_serving") and script path
    bench_prefix = bench_script.split()

    cmd = [
        sys.executable, *bench_prefix,
        "--model", model_path,
        "--backend", "sglang-oai-chat",
        "--host", host,
        "--port", str(port),
        "--num-prompts", str(num_prompts),
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def parse_benchmark_output(output: str) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


# ============================================================
# Summary
# ============================================================

COLUMNS = [
    ("Model",       "model_name",       32, None),
    ("TP",          "tp",                3, None),
    ("Status",      "status",            4, None),
    ("Req/s",       "req_throughput",    8, ".1f"),
    ("In tok/s",    "input_throughput", 10, ".1f"),
    ("Out tok/s",   "output_throughput",10, ".1f"),
    ("Tot tok/s",   "total_throughput", 10, ".1f"),
    ("E2E(ms)",     "mean_e2e_latency",  8, ".1f"),
    ("TTFT(ms)",    "mean_ttft",         8, ".1f"),
    ("P99TTFT",     "p99_ttft",          8, ".1f"),
    ("ITL(ms)",     "mean_itl",          7, ".2f"),
    ("P99ITL",      "p99_itl",           7, ".2f"),
]


def print_summary(results: List[BenchResult]) -> None:
    header = " | ".join(c[0].ljust(c[2]) for c in COLUMNS)
    sep = "-+-".join("-" * c[2] for c in COLUMNS)

    print(f"\n{'=' * len(sep)}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * len(sep)}")
    print(header)
    print(sep)

    for r in results:
        parts = []
        for _, attr, width, fmt in COLUMNS:
            val = getattr(r, attr)
            if val is None:
                text = "-"
            elif fmt:
                text = f"{val:{fmt}}"
            else:
                text = str(val)
            parts.append(text.ljust(width))
        print(" | ".join(parts))

    print(sep)
    ok = sum(1 for r in results if r.status == "OK")
    fail = sum(1 for r in results if r.status != "OK")
    print(f"  Total: {len(results)}  |  OK: {ok}  |  Failed: {fail}")
    print(f"{'=' * len(sep)}\n")


def save_csv(results: List[BenchResult], path: str) -> None:
    fieldnames = [c[1] for c in COLUMNS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({attr: getattr(r, attr) for attr in fieldnames})
    print(f"Results saved to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated benchmark runner for InfiniLM"
    )

    # Model settings
    parser.add_argument("--model-dir", type=str, default="/data/rubik/models/",
                        help="Root directory of model folders "
                             "(default: /data/rubik/models/)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Override model list with specific folder names "
                             "(e.g. --models Baichuan2-7B-Chat Meta-Llama-3-8B-Instruct). "
                             "If omitted, DEFAULT_MODEL_NAMES is used.")

    # Server settings
    parser.add_argument("--server-script", type=str,
                        default="python/infinilm/server/inference_server.py",
                        help="Path to the inference server script")
    parser.add_argument("--device", type=str, default="nvidia",
                        help="Device type passed to server (e.g. nvidia, moore)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host address")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--no-paged-attn", action="store_true",
                        help="Disable --enable-paged-attn flag for the server")
    parser.add_argument("--startup-timeout", type=int, default=300,
                        help="Max seconds to wait for server startup")

    # Benchmark settings
    parser.add_argument("--bench-script", type=str,
                        default="-m sglang.bench_serving",
                        help="Benchmark script path or module "
                             "(e.g. -m sglang.bench_serving or "
                             "/path/to/bench_serving.py)")
    parser.add_argument("--num-prompts", type=int, default=32,
                        help="Number of prompts for benchmark")
    parser.add_argument("--input-len", type=int, default=128,
                        help="Random input token length")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Random output token length")
    parser.add_argument("--bench-timeout", type=int, default=600,
                        help="Timeout in seconds for each benchmark run")

    # General settings
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Seconds to wait between models for GPU cooldown")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv",
                        help="Output CSV file path")

    args = parser.parse_args()

    # Build model name list
    model_names = args.models if args.models else DEFAULT_MODEL_NAMES

    # Build full model paths from model_dir + model_name
    model_dir = args.model_dir.rstrip("/")
    model_configs = [(build_model_path(model_dir, name), args.tp)
                     for name in model_names]

    enable_paged_attn = not args.no_paged_attn

    # Print configuration
    print(f"Configuration:")
    print(f"  Model dir:       {model_dir}")
    print(f"  Device:          {args.device}")
    print(f"  TP:              {args.tp}")
    print(f"  Host:            {args.host}")
    print(f"  Port:            {args.port}")
    print(f"  Paged Attention: {enable_paged_attn}")
    print(f"  Bench script:    {args.bench_script}")
    print(f"  Num prompts:     {args.num_prompts}")
    print(f"  Input len:       {args.input_len}")
    print(f"  Output len:      {args.output_len}")
    print(f"  Models ({len(model_configs)}):")
    for p, tp in model_configs:
        print(f"    - {p} (tp={tp})")
    print()

    # Run benchmarks
    results: List[BenchResult] = []
    total = len(model_configs)

    for idx, (model_path, tp) in enumerate(model_configs, 1):
        name = os.path.basename(model_path.rstrip("/"))
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {name}  (tp={tp})")
        print(f"{'='*60}")

        result = BenchResult(model_name=name, tp=tp)
        server_proc = None
        log_file = None
        log_path = f"/tmp/infinilm_{name}.log"

        try:
            # Start server
            print(f"  Starting server...")
            server_proc, log_file, log_path = start_server(
                model_path=model_path,
                tp=tp,
                device=args.device,
                server_script=args.server_script,
                host=args.host,
                port=args.port,
                enable_paged_attn=enable_paged_attn,
            )

            # Wait for readiness
            print(f"  Waiting for server (timeout={args.startup_timeout}s)...")
            if not wait_for_server(args.host, args.port, args.startup_timeout,
                                   proc=server_proc, log_path=log_path):
                if server_proc.poll() is not None:
                    raise RuntimeError(
                        f"Server exited prematurely (rc={server_proc.returncode}), "
                        f"see {log_path}"
                    )
                raise RuntimeError(
                    f"Server did not become ready within {args.startup_timeout}s, "
                    f"see {log_path}"
                )

            print(f"  Server is ready.")

            # Run benchmark
            print(f"  Running benchmark...")
            bench_output = run_benchmark(
                model_path=model_path,
                bench_script=args.bench_script,
                host=args.host,
                port=args.port,
                num_prompts=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                timeout=args.bench_timeout,
            )

            # Parse
            metrics = parse_benchmark_output(bench_output)
            if not any(metrics.values()):
                raise RuntimeError("No metrics parsed from benchmark output")

            for attr, val in metrics.items():
                setattr(result, attr, val)

            result.status = "OK"
            print(f"  Done: Req/s={result.req_throughput:.1f}, "
                  f"TTFT={result.mean_ttft:.1f}ms, "
                  f"ITL={result.mean_itl:.2f}ms")

        except Exception as e:
            result.status = "FAIL"
            print(f"  FAILED: {e}")

        finally:
            print(f"  Stopping server...")
            stop_server(server_proc)
            if log_file:
                log_file.close()
            # Ensure port is freed
            time.sleep(3)

        results.append(result)

        # Cooldown
        if args.cooldown > 0 and idx < total:
            print(f"  GPU cooldown: {args.cooldown}s...")
            time.sleep(args.cooldown)

    print_summary(results)
    save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

