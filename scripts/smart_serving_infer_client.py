#!/usr/bin/env python3
"""
InfiniLM Sanity Check - Client Side
====================================
Run in Terminal 2:
  python smart_serving_infer_client.py

Reads plan.json (written by the server) to learn which models to test,
then waits for each model's .ready signal (with API verification),
sends inference requests, and prints results.
"""

import argparse
import glob
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

SIGNAL_DIR = "/tmp/infinilm_signals"
PLAN_FILE = os.path.join(SIGNAL_DIR, "plan.json")

CYAN  = "\033[36m"
GREEN = "\033[32m"
RED   = "\033[31m"
BOLD  = "\033[1m"
RESET = "\033[0m"


@dataclass
class PromptResult:
    prompt: str
    response_text: Optional[str] = None
    latency_ms: Optional[float] = None
    error_msg: Optional[str] = None


@dataclass
class ModelSanityResult:
    model_name: str
    tp: int
    status: str = "PENDING"
    prompt_results: List[PromptResult] = field(default_factory=list)
    error_msg: Optional[str] = None


def _sig_path(name, suffix):
    return os.path.join(SIGNAL_DIR, name + suffix)


def _clear_sig(name, suffix):
    p = _sig_path(name, suffix)
    if os.path.exists(p):
        os.remove(p)


def _write_sig(name, suffix):
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    open(_sig_path(name, suffix), "w").close()


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def is_server_ready(host, port):
    """Check if the inference API is actually serving requests.

    Port being open is NOT enough -- uvicorn binds the port before
    the model finishes loading.  We must GET /v1/models and get a
    200 response to be sure the engine is initialized.
    """
    import urllib.request
    import urllib.error
    if not is_port_open(host, port):
        return False
    try:
        url = "http://{}:{}/v1/models".format(host, port)
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError,
            ConnectionRefusedError, socket.timeout, OSError):
        return False


def _wait_for_plan(timeout=120, poll=2):
    """Wait for the server to write plan.json."""
    dl = time.time() + timeout
    while time.time() < dl:
        if os.path.exists(PLAN_FILE):
            try:
                with open(PLAN_FILE) as f:
                    plan = json.load(f)
                if "models" in plan and len(plan["models"]) > 0:
                    return plan
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(poll)
    return None


def _wait_for_ready(name, host, port, timeout=300, poll=1):
    """Wait for .ready signal AND verify API is actually serving.

    This prevents stale signal files from a previous run from
    tricking the client into thinking the server is ready.
    """
    p = _sig_path(name, ".ready")
    dl = time.time() + timeout
    while time.time() < dl:
        if os.path.exists(p):
            if is_server_ready(host, port):
                return True
            else:
                print("  (stale .ready detected, API not serving, clearing it)")
                _clear_sig(name, ".ready")
        time.sleep(poll)
    return False


def send_prompt(host, port, model_name, prompt, timeout, max_tokens=128):
    import urllib.request
    import urllib.error

    res = PromptResult(prompt=prompt)
    url = "http://{}:{}/v1/chat/completions".format(host, port)
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
        res.latency_ms = (time.time() - t0) * 1000
        data = json.loads(body)
        res.response_text = data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        b = e.read().decode(errors="replace")
        res.error_msg = "HTTP {}: {}".format(e.code, b[:300])
    except Exception as e:
        res.error_msg = str(e)
    return res


def _truncate(text, n=80):
    if not text:
        return "-"
    if n == 0:
        return text
    t = text.strip().replace("\n", " ")
    return t[:n] + "..." if len(t) > n else t


def print_summary(results, display_len=0):
    print()
    s = "=" * 70
    print(BOLD + s + RESET)
    print(BOLD + "  SANITY CHECK SUMMARY" + RESET)
    print(BOLD + s + RESET)
    print()

    for r in results:
        if r.status == "OK":
            ic = GREEN + "[OK]" + RESET
        else:
            ic = RED + "[FAIL]" + RESET
        print("  " + ic + " " + r.model_name + " (tp=" + str(r.tp) + ")")
        if r.error_msg:
            print("       Error: " + r.error_msg[:120])
        for pr in r.prompt_results:
            p = _truncate(pr.prompt, 30)
            if pr.error_msg:
                print("       [" + p + "] -> " + _truncate(pr.error_msg, 60))
            elif pr.response_text:
                lat = "{:.0f}ms".format(pr.latency_ms) if pr.latency_ms else "-"
                print("       [" + p + "] (" + lat + ") -> " + _truncate(pr.response_text, display_len))

    ok = sum(1 for r in results if r.status == "OK")
    fail = len(results) - ok
    print()
    print("  Total: {} | ".format(len(results)) +
          GREEN + "OK: " + str(ok) + RESET + " | " +
          RED + "Failed: " + str(fail) + RESET)
    print()


def main():
    ap = argparse.ArgumentParser(description="InfiniLM client-side runner")
    ap.add_argument("--prompts", nargs="+",
                    default=["What is artificial intelligence?",
                             "Please introduce AI in one sentence."])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--request-timeout", type=int, default=120)
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--plan-timeout", type=int, default=120,
                    help="Seconds to wait for server to write plan.json")
    ap.add_argument("--display-len", type=int, default=0,
                    help="Max chars to display per response (0=show all)")
    args = ap.parse_args()

    # ---- Read plan from server ----
    print("Waiting for server to publish plan ...")
    plan = _wait_for_plan(timeout=args.plan_timeout)
    if plan is None:
        print(RED + "ERROR: Timed out waiting for " + PLAN_FILE + RESET)
        print("  Make sure the server script is running first!")
        sys.exit(1)

    model_names = plan["models"]
    tp = plan["tp"]
    host = plan["host"]
    port = plan["port"]

    print(GREEN + "Got plan from server:" + RESET)
    print("  models: {}".format(model_names))
    print("  tp: {}".format(tp))
    print("  host: {}".format(host))
    print("  port: {}".format(port))
    print()

    total = len(model_names)
    results = []

    for idx, name in enumerate(model_names, 1):
        print()
        print(BOLD + "=" * 70 + RESET)
        print(BOLD + "  [{}/{}]  {}  (tp={})".format(idx, total, name, tp) + RESET)
        print(BOLD + "=" * 70 + RESET)
        print()
        print("异构智算推理系统-服务感知的推理系统")
        print("大模型推理 国产GPU卡 & 模型适配种类测试")
        print()
        print()

        print("  Waiting for server to start ...")
        if not _wait_for_ready(name, host, port, timeout=args.ready_timeout):
            print(RED + "  TIMEOUT: Server never became ready" + RESET)
            result = ModelSanityResult(
                model_name=name, tp=tp, status="FAIL",
                error_msg="Server never became ready within {}s".format(
                    args.ready_timeout))
            results.append(result)
            _write_sig(name, ".done")
            continue

        print(GREEN + "  Server is READY!" + RESET)
        print()

        result = ModelSanityResult(model_name=name, tp=tp)
        all_ok = True
        for prompt in args.prompts:
            print("  " + CYAN + ">> Request :" + RESET + " " + repr(prompt))
            pr = send_prompt(host, port, name, prompt,
                             args.request_timeout, args.max_tokens)
            result.prompt_results.append(pr)
            if pr.error_msg:
                print("  " + RED + "<< Error   :" + RESET + " " + pr.error_msg[:200])
                all_ok = False
            else:
                ms = "{:.0f}".format(pr.latency_ms)
                short = _truncate(pr.response_text, args.display_len)
                print("  " + GREEN + "<< Response:" + RESET +
                      " (" + ms + "ms) " + short)
            time.sleep(1)

        result.status = "OK" if all_ok else "FAIL"
        results.append(result)

        _write_sig(name, ".done")
        print()
        if result.status == "OK":
            print("  " + GREEN + "OK" + RESET + " " + name +
                  " -> " + result.status)
        else:
            print("  " + RED + "FAIL" + RESET + " " + name +
                  " -> " + result.status)

    print_summary(results, display_len=args.display_len)

    if any(r.status != "OK" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

