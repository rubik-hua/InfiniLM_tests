#!/usr/bin/env python3
"""
Sanity check runner for InfiniLM.
Starts the server for each model, sends one or more prompts via the
OpenAI-compatible API, and verifies that the inference service works correctly.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# Default model names
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
class PromptResult:
    """Stores the result of a single prompt inference."""
    prompt: str
    response_text: Optional[str] = None
    latency_ms: Optional[float] = None
    error_msg: Optional[str] = None

@dataclass
class ModelSanityResult:
    """Stores the overall infer check result for a model."""
    model_name: str
    tp: int
    status: str = "PENDING"  # OK, FAIL
    prompt_results: List[PromptResult] = field(default_factory=list)
    error_msg: Optional[str] = None  # For server-level errors (e.g., crash)

# ============================================================
# Helpers: server management
# ============================================================

def build_model_path(model_dir: str, model_name: str) -> str:
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
            time.sleep(2)  # Extra grace period
            return True
        time.sleep(3)
    return False


def start_server(model_path: str, tp: int, device: str, server_script: str,
                 host: str, port: int, enable_paged_attn: bool = True):
    env = os.environ.copy()
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
    log_path = f"/tmp/infinilm_infer_{name}.log"
    log_file = open(log_path, "w")

    print(f"  Server log: {log_path}")
    print(f"  Server cmd: {' '.join(cmd)}")

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

# ============================================================
# Helpers: send prompt and check response
# ============================================================

def send_prompt(host: str, port: int, model_name: str,
                prompt: str, timeout: int,
                max_tokens: int = 128,
                temperature: float = 0.0) -> PromptResult:
    """
    Send a chat completion request via the OpenAI-compatible API
    using urllib (no extra dependencies needed).
    """
    import urllib.request
    import urllib.error

    result = PromptResult(prompt=prompt)
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result.latency_ms = (time.time() - start) * 1000

        data = json.loads(body)
        result.response_text = data["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        result.error_msg = f"HTTP {e.code}: {body[:300]}"
    except Exception as e:
        result.error_msg = str(e)

    return result

# ============================================================
# Summary
# ============================================================

def truncate_text(text: str, max_len: int = 80) -> str:
    if not text:
        return "-"
    text = text.strip().replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def print_summary(results: List[ModelSanityResult]) -> None:
    print()
    print("=" * 90)
    print("  SANITY CHECK SUMMARY")
    print("=" * 90)

    for r in results:
        status_icon = "✅" if r.status == "OK" else "❌"
        print(f"\n  {status_icon}  {r.model_name:<38s}  TP={r.tp}  Status={r.status}")

        if r.error_msg:
            print(f"     Server Error: {truncate_text(r.error_msg, 120)}")

        for p_res in r.prompt_results:
            p_short = truncate_text(p_res.prompt, 30)
            if p_res.error_msg:
                print(f"     [{p_short}] -> Error: {truncate_text(p_res.error_msg, 100)}")
            elif p_res.response_text:
                lat = f"{p_res.latency_ms:.0f}ms" if p_res.latency_ms else "-"
                resp_short = truncate_text(p_res.response_text, 80)
                print(f"     [{p_short}] ({lat}) -> {resp_short}")

    # Final tally
    print()
    ok = sum(1 for r in results if r.status == "OK")
    fail = sum(1 for r in results if r.status != "OK")
    print("-" * 90)
    print(f"  Total: {len(results)}  |  OK: {ok}  |  Failed: {fail}")
    print("=" * 90)
    print()

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check runner for InfiniLM inference service"
    )

    # Model settings
    parser.add_argument("--model-dir", type=str, default="/data/rubik/models/",
                        help="Root directory of model folders")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Override model list. If omitted, DEFAULT_MODEL_NAMES is used.")

    # Server settings
    parser.add_argument("--server-script", type=str,
                        default="python/infinilm/server/inference_server.py",
                        help="Path to the inference server script")
    parser.add_argument("--device", type=str, default="nvidia",
                        help="Device type (e.g. nvidia, moore)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallelism degree")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host address")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--no-paged-attn", action="store_true",
                        help="Disable --enable-paged-attn flag")
    parser.add_argument("--startup-timeout", type=int, default=300,
                        help="Max seconds to wait for server startup")

    # Prompt settings (Support multiple prompts)
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=[
                            "What is artificial intelligence?",
                            "请用一句话介绍人工智能。"
                        ],
                        help="One or more test prompts. Defaults to one English and one Chinese.")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens for the response")
    parser.add_argument("--request-timeout", type=int, default=120,
                        help="Timeout in seconds for each inference request")

    # General settings
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Seconds to wait between models for GPU cooldown")

    args = parser.parse_args()

    # Build model list
    model_names = args.models if args.models else DEFAULT_MODEL_NAMES
    model_dir = args.model_dir.rstrip("/")
    model_configs = [(build_model_path(model_dir, name), args.tp)
                     for name in model_names]
    enable_paged_attn = not args.no_paged_attn

    # Print configuration
    print("Configuration:")
    print(f"  Model dir:       {model_dir}")
    print(f"  Device:          {args.device}")
    print(f"  TP:              {args.tp}")
    print(f"  Host:            {args.host}")
    print(f"  Port:            {args.port}")
    print(f"  Paged Attention: {enable_paged_attn}")
    print(f"  Prompts ({len(args.prompts)}):")
    for i, p in enumerate(args.prompts, 1):
        print(f"    {i}. {p}")
    print(f"  Max tokens:      {args.max_tokens}")
    print(f"  Request timeout: {args.request_timeout}s")
    print(f"  Models ({len(model_configs)}):")
    for p, tp in model_configs:
        print(f"    - {p} (tp={tp})")
    print()

    # Run infer checks
    results: List[ModelSanityResult] = []
    total = len(model_configs)

    for idx, (model_path, tp) in enumerate(model_configs, 1):
        name = os.path.basename(model_path.rstrip("/"))
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {name}  (tp={tp})")
        print(f"{'='*60}")

        result = ModelSanityResult(model_name=name, tp=tp)
        server_proc = None
        log_file = None
        log_path = f"/tmp/infinilm_infer_{name}.log"

        try:
            # 1. Start server
            print("  Starting server...")
            server_proc, log_file, log_path = start_server(
                model_path=model_path,
                tp=tp,
                device=args.device,
                server_script=args.server_script,
                host=args.host,
                port=args.port,
                enable_paged_attn=enable_paged_attn,
            )

            # 2. Wait for readiness
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
            print("  Server is ready.")

            # 3. Send prompts sequentially
            all_prompts_ok = True
            for prompt in args.prompts:
                print(f"  Sending prompt: \"{truncate_text(prompt, 40)}\"")
                p_res = send_prompt(
                    host=args.host,
                    port=args.port,
                    model_name=name,
                    prompt=prompt,
                    timeout=args.request_timeout,
                    max_tokens=args.max_tokens,
                )
                result.prompt_results.append(p_res)

                if p_res.error_msg:
                    print(f"    Failed: {p_res.error_msg}")
                    all_prompts_ok = False
                else:
                    print(f"    Response ({p_res.latency_ms:.0f}ms): {p_res.response_text}")

            result.status = "OK" if all_prompts_ok else "FAIL"

        except Exception as e:
            result.status = "FAIL"
            result.error_msg = str(e)
            print(f"  FAILED: {e}")

        finally:
            print("  Stopping server...")
            stop_server(server_proc)
            if log_file:
                log_file.close()
            time.sleep(3)

        results.append(result)

        # Cooldown between models
        if args.cooldown > 0 and idx < total:
            print(f"  GPU cooldown: {args.cooldown}s...")
            time.sleep(args.cooldown)

    # Print summary
    print_summary(results)

    # Exit with non-zero code if any model failed
    if any(r.status != "OK" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

