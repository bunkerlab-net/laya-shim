"""Serves a Laya checkpoint on TypeSafe's System One route, POST /v1/systemone.

omp's `typesafe` API sends {"state", "model", "questions"} and reads back
{"model", "answers", "usage"}. Laya's predict() takes and returns that shape
already, so this server only adds HTTP. It ignores the request's "model" field
and always answers with the one checkpoint it loaded at startup.

Environment variables:
  LAYA_BACKEND    `mlx` for Laya-MLX or `torch` for upstream Laya. Default: mlx.
  LAYA_MODEL      Hugging Face repository or local path.
                  Default: convaiinnovations/laya.
  LAYA_SUBFOLDER  Checkpoint inside LAYA_MODEL: empty for English,
                  `multilingual`, or `typed-decisions`. Default: typed-decisions.
  LAYA_HOST       Bind address. Default: 127.0.0.1.
  LAYA_PORT       Bind port. Default: 8765.
  LAYA_LOG_COLOR  `auto` colors log levels when standard error is a terminal
                  and NO_COLOR isn't set. `always` or `never` overrides that.
                  Default: auto.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

BACKEND = os.environ.get("LAYA_BACKEND", "mlx")
MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", "typed-decisions") or None
HOST = os.environ.get("LAYA_HOST", "127.0.0.1")
PORT = int(os.environ.get("LAYA_PORT", "8765"))
LOG_COLOR = os.environ.get("LAYA_LOG_COLOR", "auto")

log = logging.getLogger("laya-shim")

LEVEL_COLORS = {
    logging.DEBUG: "2",
    logging.INFO: "32",
    logging.WARNING: "33",
    logging.ERROR: "31",
    logging.CRITICAL: "1;31",
}


class ColorFormatter(logging.Formatter):
    """Colors the level name with an ANSI escape code."""

    def formatMessage(self, record):
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = (
            f"\033[{LEVEL_COLORS[record.levelno]}m{record.levelname}\033[0m"
        )
        return super().formatMessage(record)


def use_color():
    if LOG_COLOR == "always":
        return True
    if LOG_COLOR == "never":
        return False
    if LOG_COLOR != "auto":
        raise SystemExit(
            f"LAYA_LOG_COLOR must be auto, always, or never, got {LOG_COLOR!r}"
        )
    return sys.stderr.isatty() and "NO_COLOR" not in os.environ


def log_warning(message, category, filename, lineno, file=None, line=None):
    # Replaces Python's default warning output, which prints the file path
    # and the source line that raised the warning.
    logging.getLogger("warnings").warning("%s", message)


def configure_logging():
    handler = logging.StreamHandler()
    formatter = ColorFormatter if use_color() else logging.Formatter
    handler.setFormatter(formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    warnings.showwarning = log_warning
    # The Hugging Face client logs every request through httpx. A HEAD request
    # checks the local cache and a GET request downloads a file, so keep only
    # the GET lines. The client draws its own progress bar for each file.
    logging.getLogger("httpx").addFilter(
        lambda record: record.getMessage().startswith("HTTP Request: GET")
    )


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.started = time.perf_counter()
        self.questions = 0
        if self.path != "/v1/systemone":
            self.reply(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.questions = len(body["questions"])
            result = self.server.agent.predict(body["state"], body["questions"])
        except (KeyError, TypeError, ValueError) as e:
            # Bad JSON, a missing field, or a question Laya rejects, such as
            # too many choice options for its token budget. omp doesn't retry
            # a 4xx and moves on to the next judge in its fallback chain.
            reason = f"missing field {e}" if isinstance(e, KeyError) else str(e)
            log.warning("rejected request: %s", reason)
            self.reply(422, {"error": reason})
            return
        except Exception:
            log.exception("inference failed")
            self.reply(500, {"error": "inference failed"})
            return
        self.reply(200, result)

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log.info(
            '%s "%s %s" %d questions=%d %.1f ms',
            self.client_address[0],
            self.command,
            self.path,
            status,
            self.questions,
            (time.perf_counter() - self.started) * 1000,
        )

    def log_request(self, code="-", size="-"):
        # reply() writes the access log line, with timing.
        pass

    def log_message(self, format, *args):
        log.warning("%s %s", self.client_address[0], format % args)


def load_agent():
    """Loads the checkpoint that LAYA_BACKEND, LAYA_MODEL, and LAYA_SUBFOLDER name."""
    if BACKEND == "mlx":
        import laya_mlx as laya
    elif BACKEND == "torch":
        import laya
    else:
        raise SystemExit(f"LAYA_BACKEND must be mlx or torch, got {BACKEND!r}")
    return laya.load(MODEL, subfolder=SUBFOLDER)


def serve():
    configure_logging()
    log.info(
        "starting: backend=%s model=%s subfolder=%s",
        BACKEND,
        MODEL,
        SUBFOLDER or "(root)",
    )
    log.info("loading checkpoint; the first run downloads it from Hugging Face")
    started = time.perf_counter()
    agent = load_agent()
    log.info("checkpoint loaded in %.1f s", time.perf_counter() - started)

    # HTTPServer handles one request at a time, so only one forward pass runs
    # at once. A decision takes tens of milliseconds, so requests rarely wait.
    server = HTTPServer((HOST, PORT), Handler)
    server.agent = agent

    def stop(signum, frame):
        log.info(
            "received %s, finishing the current request and stopping",
            signal.Signals(signum).name,
        )
        # shutdown() waits for serve_forever() to return, and serve_forever()
        # runs on this thread, so call it from another thread.
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    log.info("listening on http://%s:%d/v1/systemone", HOST, PORT)
    server.serve_forever()
    server.server_close()
    log.info("stopped")


def main():
    """Entry point for the `laya-shim` command."""
    try:
        serve()
    except KeyboardInterrupt:
        # Ctrl+C before the server starts, for example during a download.
        log.info("interrupted during startup")
        sys.exit(130)
