# laya-shim

`laya_shim.py` runs a [Laya](https://huggingface.co/convaiinnovations/laya)
checkpoint behind TypeSafe's System One route, `POST /v1/systemone`. That
route is the one omp calls for its `judge` model role. Point the role at this
server and omp's typed yes/no, choice, and score decisions run on your machine
instead of on TypeSafe's Jev.

The shim runs one of two backends:

| `LAYA_BACKEND`  | Library                                                                     | Runs on                       |
| --------------- | --------------------------------------------------------------------------- | ----------------------------- |
| `mlx` (default) | [Laya-MLX](https://github.com/mizorewww/laya-mlx), an MLX port              | Apple silicon GPU             |
| `torch`         | [Laya](https://github.com/NandhaKishorM/laya), the upstream PyTorch release | Any platform PyTorch supports |

Both libraries load the same Hugging Face checkpoints, and their `predict()`
functions take and return the same JSON. You can switch backends without
changing omp's config.

Upstream Laya also ships its own Jev-compatible server, `laya-serve`. This shim
exists so that both backends run behind one server with one set of settings.

## Before you begin

Install [mise](https://mise.jdx.dev). mise installs uv, and uv installs Python
and the backend's packages the first time you start the server.

## Start the server

```sh
mise run serve
```

On the first run, the server downloads the checkpoint from Hugging Face. The
typed-decisions checkpoint is about 850 MB. Later runs load it from the local
cache. On an Apple silicon Mac, that took about 1 second with MLX and 18
seconds with PyTorch. The server is ready when it logs this line:

```text
INFO laya-shim: listening on http://127.0.0.1:8765/v1/systemone
```

To check it by hand, send a request:

```sh
curl -s localhost:8765/v1/systemone -d '{
  "state": "All tests pass now.",
  "questions": {"claims": {"type": "noul", "instructions": "Does the reply claim tests pass?"}}
}'
```

To stop the server, press Ctrl+C or send it `SIGTERM`. The server finishes the
request it's answering, closes its socket, and logs `stopped`. After Ctrl+C,
`mise run` exits with status 130, the usual status for an interrupted command.
If you press Ctrl+C during startup, for example during a download, the server
logs `interrupted during startup` and exits.

## Configure the server

`mise.toml` sets these variables. The shim uses the same defaults when you run
it without mise.

| Variable         | Default                  | Meaning                                                                                                                        |
| ---------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `LAYA_BACKEND`   | `mlx`                    | `mlx` or `torch`.                                                                                                              |
| `LAYA_MODEL`     | `convaiinnovations/laya` | Hugging Face repository or local path.                                                                                         |
| `LAYA_SUBFOLDER` | `typed-decisions`        | Checkpoint inside `LAYA_MODEL`. Leave it empty for English, or set `multilingual` or `typed-decisions`.                        |
| `LAYA_HOST`      | `127.0.0.1`              | Bind address.                                                                                                                  |
| `LAYA_PORT`      | `8765`                   | Bind port.                                                                                                                     |
| `LAYA_LOG_COLOR` | `auto`                   | `auto` colors log levels when standard error is a terminal and `NO_COLOR` isn't set. Set `always` or `never` to override that. |

To change a value for this checkout only, create `mise.local.toml`. Git ignores
this file.

```toml
[env]
LAYA_BACKEND = "torch"
```

To change a value for one run, set it in your shell:

```sh
LAYA_BACKEND=torch mise run serve
```

A value from your shell or from `mise.local.toml` takes precedence over the
default in `mise.toml`.

When you switch backends, uv removes the other backend's packages and installs
the ones you selected. Both backends read the same Hugging Face cache, so
switching doesn't download the checkpoint again.

### Choose a checkpoint

| `LAYA_SUBFOLDER`  | Context      | Notes                                                                                                                             |
| ----------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| (empty)           | 512 tokens   | English only.                                                                                                                     |
| `multilingual`    | 1,024 tokens | More than 100 languages.                                                                                                          |
| `typed-decisions` | 1,024 tokens | Fine-tuned on typed decisions. On upstream's typed-decisions benchmark it scores 0.766, against 0.362 for the English checkpoint. |

## Connect omp

1.  Add a provider to `~/.omp/agent/models.yml`:

    ```yaml
    providers:
      laya:
        baseUrl: http://127.0.0.1:8765
        api: typesafe
        apiKey: laya-local
        models:
          - id: laya-local
            name: Laya (local)
    ```

    `api: typesafe` makes omp send judge requests to
    `{baseUrl}/v1/systemone`. omp skips a judge that has no API key, so
    `apiKey` needs a value. The shim ignores it.

1.  Select the model for the `judge` role in `~/.omp/agent/config.yml`:

    ```yaml
    modelRoles:
      judge: laya/laya-local
    retry:
      fallbackChains:
        judge:
          - typesafe/jev-latest
    ```

    If the shim is down or rejects a request, omp tries the next judge in
    `fallbackChains.judge`. To keep every judgment on your machine, set
    `judge: []` instead.

1.  Start the server, and then run a command that uses the judge:

    ```sh
    omp find "where the shim returns HTTP 422" .
    ```

    Each judgment adds an access log line to the server's output:

    ```text
    INFO laya-shim: 127.0.0.1 "POST /v1/systemone" 200 questions=3 352.5 ms
    ```

omp treats any model with `api: typesafe` as a native System One judge, the
same as Jev. Two settings that default to `auto` turn on because of this:

- `find.enabled` adds the `find` tool.
- `ttsr.judge` asks judged rulebook rules about each finished reply and tool
  call.

## Limits

These limits come from Laya, not from the shim. Check them before you rely on
the answers.

- **Short context.** Jev reads about 33,000 tokens. Laya reads 512 or 1,024,
  and that budget covers the question, the options, and the state. Laya cuts
  the end off a longer state without an error. omp sends long states for
  judged rulebook rules, git staging, and `find`, so in those cases Laya only
  sees the start. If judged rules give bad answers, set `ttsr.judge: off`.
- **Few choice options.** All options for one question share a budget of 192
  to 256 tokens. A question with too many options fails, and the shim returns
  `422`. omp doesn't retry a `422` and moves on to the next judge in the chain.
- **Weak yes/no answers.** Upstream reports that `noul` questions can follow the
  option labels instead of the input, most often on the English checkpoint. In
  a test with the typed-decisions checkpoint, "All tests pass now." scored 0.54
  for "Does the reply claim tests pass?", which is close to a coin toss.
- **Latency.** Upstream's figures are for short inputs: 13 ms on an M3 Max with
  MLX, and 33 to 40 ms on a T4 GPU with PyTorch. With omp's `find`, which sends
  whole files, each request took 200 to 470 ms on MLX and 80 to 1,140 ms on
  PyTorch on an Apple silicon Mac.

## Logs

The server writes these log lines to standard error:

- Startup settings, and when the checkpoint starts and finishes loading.
- One `httpx` line for each GET request to Hugging Face, which includes file
  downloads. The Hugging Face client also draws a progress bar for each file.
- Warnings from Laya, one line each, such as the calibration temperature it
  clamps at load time.
- One access log line for each request, with the status, question count, and
  time.
- The reason for each rejected request, and a traceback for each failed
  inference.

When color is on, `INFO` is green, `WARNING` is yellow, and `ERROR` is red.

## Development

[hk](https://hk.jdx.dev) runs `ruff check` and then `ruff format` as a
pre-commit hook. mise installs the hook when a shell with `mise activate`
enters this directory. To install it by hand, run `hk install --mise`. The hook
fixes what it can and stages the result. It blocks the commit if an error
remains that ruff can't fix. To skip it for one commit, run
`HK=0 git commit`.

To run the same checks on every file, run `hk check --all`. To apply the fixes,
run `hk fix --all`.
