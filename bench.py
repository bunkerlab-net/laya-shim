"""Measures request latency against a running laya_shim.py server.

Each workload sends the same System One request `--warmup` times without
timing it, then `--iterations` times with timing. The time covers the whole
HTTP round trip on this machine, so it includes JSON encoding and the local
socket as well as inference.

Start the server first with `mise run serve`, and then run `mise run bench`.
The script reads LAYA_HOST and LAYA_PORT to find the server.
"""

import argparse
import json
import os
import statistics
import time
import urllib.request

NOUL = {"type": "noul", "instructions": "Does the reply claim that tests pass?"}
SCORE = {
    "type": "score",
    "instructions": "How hard is this task?",
    "criteria": ["easy", "medium", "hard"],
}
CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs and outages",
        "sales": "new purchases",
    },
}
PARAGRAPH = (
    "The build finished on the second attempt after the cache was cleared. "
    "Two tests in the parser module still fail on Windows because of path "
    "separators, and the reviewer asked for a regression test before merging. "
)

# The long state is about 12,700 characters. Laya reads at most 1,024 tokens
# and drops the rest, so this workload measures the cost of a full context.
WORKLOADS = {
    "short, 1 question": ("All tests pass now.", {"q": NOUL}),
    "short, 3 questions": (
        "I was billed twice. Please refund the duplicate today.",
        {"noul": NOUL, "score": SCORE, "choice": CHOICE},
    ),
    "short, 10 questions": (
        "All tests pass now.",
        {f"q{i}": NOUL for i in range(10)},
    ),
    "long, 1 question": (PARAGRAPH * 60, {"q": NOUL}),
}


def post(url, state, questions):
    body = json.dumps({"model": "bench", "state": state, "questions": questions})
    request = urllib.request.Request(
        url, data=body.encode(), headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        response.read()
    return (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    host = os.environ.get("LAYA_HOST", "127.0.0.1")
    port = os.environ.get("LAYA_PORT", "8765")
    url = f"http://{host}:{port}/v1/systemone"
    print(f"{url}, {args.warmup} warmup and {args.iterations} timed requests each\n")
    print(f"{'workload':<22} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8} {'q/s':>8}")

    for name, (state, questions) in WORKLOADS.items():
        for _ in range(args.warmup):
            post(url, state, questions)
        times = sorted(post(url, state, questions) for _ in range(args.iterations))
        p95 = times[min(len(times) - 1, round(0.95 * len(times)) - 1)]
        mean = statistics.fmean(times)
        rate = len(questions) * 1000 / mean
        print(
            f"{name:<22} {statistics.median(times):>8.1f} {p95:>8.1f}"
            f" {mean:>8.1f} {rate:>8.1f}"
        )


if __name__ == "__main__":
    main()
