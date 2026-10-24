"""
Warm, one-request-at-a-time latency bench, separate from selfcheck.py so you
can iterate on latency without re-scoring quality every time.

    python bench.py --url http://localhost:8000/v1/decide

reports p50/p95 per decision type against an already-running server.

    python bench.py --sweep --threads 1,2,4,8

spawns its own `uvicorn app:app` subprocess once per thread count (setting
ORT_INTRA_THREADS, read by encoder.py at startup), benchmarks each, and
prints which setting wins on this machine. Needs data/dev.jsonl for request
bodies; does not touch the server's model weights.
"""

import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse

DECISIONS = ["route", "approve", "grounded"]


def percentile(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Client:
    def __init__(self, url, timeout=10.0):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.path = u.path or "/"
        self.timeout = timeout
        self.conn = None

    def call(self, payload):
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)), "Connection": "keep-alive"}
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        t0 = time.perf_counter()
        self.conn.request("POST", self.path, body=body, headers=headers)
        r = self.conn.getresponse()
        r.read()
        return (time.perf_counter() - t0) * 1000.0


def load_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def run_bench(url, rows, warmup, n_per_decision):
    client = Client(url)
    for i in range(warmup):
        r = rows[i % len(rows)]
        client.call({"request_id": r["request_id"], "decision": DECISIONS[i % 3],
                     "state": r["state"]})

    lat = {d: [] for d in DECISIONS}
    i = 0
    for d in DECISIONS:
        for k in range(n_per_decision):
            r = rows[(i + k) % len(rows)]
            ms = client.call({"request_id": r["request_id"], "decision": d,
                               "state": r["state"]})
            lat[d].append(ms)
        i += n_per_decision
    return {d: (percentile(lat[d], 0.5), percentile(lat[d], 0.95)) for d in DECISIONS}


def print_report(label, report):
    mean_p95 = sum(v[1] for v in report.values()) / 3.0
    print("\n%s" % label)
    for d in DECISIONS:
        p50, p95 = report[d]
        print("  %-9s p50 %6.2f ms   p95 %6.2f ms" % (d, p50, p95))
    print("  mean p95 across decisions: %6.2f ms" % mean_p95)
    return mean_p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/decide")
    ap.add_argument("--eval", default="data/dev.jsonl")
    ap.add_argument("--n", type=int, default=150, help="calls per decision type")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--sweep", action="store_true",
                     help="spawn a server per thread count and compare")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--port", type=int, default=8321)
    args = ap.parse_args()

    rows = load_rows(args.eval, max(args.n, 50))

    if not args.sweep:
        report = run_bench(args.url, rows, args.warmup, args.n)
        print_report("latency against %s" % args.url, report)
        return

    results = {}
    for t in [int(x) for x in args.threads.split(",")]:
        env = os.environ.copy()
        env["ORT_INTRA_THREADS"] = str(t)
        url = "http://127.0.0.1:%d/v1/decide" % args.port
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
             "--port", str(args.port)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            health_url = "http://127.0.0.1:%d/health" % args.port
            u = urllib.parse.urlparse(health_url)
            for _ in range(60):
                try:
                    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1.0)
                    c.request("GET", u.path)
                    if c.getresponse().status == 200:
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                print("server did not start for threads=%d" % t)
                continue
            report = run_bench(url, rows, args.warmup, args.n)
            mean_p95 = print_report("ORT_INTRA_THREADS=%d" % t, report)
            results[t] = mean_p95
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            time.sleep(1)

    if results:
        best = min(results, key=results.get)
        print("\nbest setting: ORT_INTRA_THREADS=%d  (mean p95 %.2f ms)" % (best, results[best]))
        print("set this before starting the server, e.g.:")
        print("  ORT_INTRA_THREADS=%d uvicorn app:app --host 0.0.0.0 --port 8000" % best)


if __name__ == "__main__":
    main()
