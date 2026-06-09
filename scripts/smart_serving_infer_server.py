#!/usr/bin/env python3
"""
InfiniLM Sanity Check - Server Side
====================================
Run in Terminal 1:
  python smart_serving_infer_server.py --device nvidia --tp 2 --models DeepSeek-R1-Distill-Qwen-7B

At startup, cleans up old signals and writes plan.json for the client.
Then for each model: start server -> signal .ready -> wait .done -> stop.
"""

import argparse
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time
import threading

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

SIGNAL_DIR = "/tmp/infinilm_signals"
PLAN_FILE = os.path.join(SIGNAL_DIR, "plan.json")


def _sig_path(name, suffix):
    return os.path.join(SIGNAL_DIR, name + suffix)


def _write_sig(name, suffix):
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    open(_sig_path(name, suffix), "w").close()


def _wait_sig(name, suffix, timeout=600, poll=1):
    p = _sig_path(name, suffix)
    dl = time.time() + timeout
    while time.time() < dl:
        if os.path.exists(p):
            return True
        time.sleep(poll)
    return False


def _clear_sig(name, suffix):
    p = _sig_path(name, suffix)
    if os.path.exists(p):
        os.remove(p)

def kill_port(port):
    """Kill any leftover process on the port before starting new server."""
    os.system("fuser -k {}/tcp 2>/dev/null || true".format(port))
    # Wait for port to be fully released
    for _ in range(10):
        if not is_port_open("127.0.0.1", port):
            return
        time.sleep(1)


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


def _reader(stream):
    for raw in iter(stream.readline, b""):
        print(raw.decode(errors="replace"), end="", flush=True)
    stream.close()


def main():
    ap = argparse.ArgumentParser(description="InfiniLM server-side runner")
    ap.add_argument("--model-dir", default="/data/rubik/models/")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--server-script",
                    default="python/infinilm/server/inference_server.py")
    ap.add_argument("--device", default="nvidia")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-paged-attn", action="store_true")
    ap.add_argument("--startup-timeout", type=int, default=300)
    ap.add_argument("--done-timeout", type=int, default=600)
    ap.add_argument("--cooldown", type=int, default=5)
    args = ap.parse_args()

    model_names = args.models or DEFAULT_MODEL_NAMES
    model_dir = args.model_dir.rstrip("/")
    enable_pa = not args.no_paged_attn
    total = len(model_names)

    # ---- Clean up ALL old signals ----
    if os.path.exists(SIGNAL_DIR):
        for f in glob.glob(os.path.join(SIGNAL_DIR, "*")):
            os.remove(f)
    os.makedirs(SIGNAL_DIR, exist_ok=True)

    # ---- Write plan.json for client ----
    plan = {
        "models": model_names,
        "tp": args.tp,
        "host": args.host,
        "port": args.port,
        "timestamp": time.time(),
    }
    with open(PLAN_FILE, "w") as f:
        json.dump(plan, f, indent=2)
    print("Wrote plan: " + PLAN_FILE)
    print("  models: {}".format(model_names))
    print("  tp: {}".format(args.tp))
    print("  host: {}".format(args.host))
    print("  port: {}".format(args.port))
    print()

    # ---- Main loop ----
    for idx, name in enumerate(model_names, 1):
        model_path = os.path.join(model_dir, name)

        print()
        print()
        print("InfiniLM Inference Server")
        print("Starting model: " + name)
        print()
        print("=" * 70)
        print("  [{}/{}]  {}  (tp={})".format(idx, total, name, args.tp))
        print("=" * 70)
        print()

        _clear_sig(name, ".ready")
        _clear_sig(name, ".done")
        kill_port(args.port)

        cmd = [
            sys.executable, args.server_script,
            "--device", args.device,
            "--model=" + model_path,
            "--tp=" + str(args.tp),
            "--host", args.host,
            "--port", str(args.port),
        ]
        if enable_pa:
            cmd.append("--enable-paged-attn")

        print("异构智算推理系统")
        print("版本：V1.0")
        print("运行状态：就绪")
        print()
        print()
        print("大模型推理 国产GPU卡 & 模型适配种类测试")
        print()
        print("CMD:", " ".join(cmd))
        print()

        proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        t = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        t.start()

        print("--- Waiting for server to start (timeout={}s) ---".format(
            args.startup_timeout))
        print()
        print("基于 " + "异构智算统一架构 V0.1" + "  推理引擎" + ", 开始加载...")
        print()
        dl = time.time() + args.startup_timeout
        ready = False
        while time.time() < dl:
            if proc.poll() is not None:
                print("ERROR: Server died (rc={})".format(proc.returncode))
                break
            if is_server_ready(args.host, args.port):
                print("--- Server is READY ---")
                sys.stdout.flush()
                ready = True
                break
            time.sleep(3)

        if ready:
            _write_sig(name, ".ready")
            print("--- Signaled client: .ready ---")
            print("--- Waiting for client to finish ---")
            if _wait_sig(name, ".done", timeout=args.done_timeout):
                print("--- Client finished ---")
            else:
                print("--- TIMEOUT waiting for client ---")
        else:
            if proc.poll() is None:
                print("--- Server startup FAILED (timeout) ---")

        print("--- Stopping server ---")
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
        print("--- Server stopped ---")

        if args.cooldown > 0 and idx < total:
            print("--- Cooldown {}s ---".format(args.cooldown))
            time.sleep(args.cooldown)

    _write_sig("all", ".done")
    print()
    print("=" * 70)
    print("  ALL MODELS DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

