// The samples are the omp requests in tests/fixtures, which the contract
// tests also send. Each fixture's `source` names the omp file it copies.
import autoThinking from "../tests/fixtures/auto-thinking.json";
import evalArrayState from "../tests/fixtures/eval-array-state.json";
import evalTextState from "../tests/fixtures/eval-text-state.json";
import findNames from "../tests/fixtures/find-names.json";
import findPassages from "../tests/fixtures/find-passages.json";
import findSketches from "../tests/fixtures/find-sketches.json";
import gitStageUnit from "../tests/fixtures/git-stage-unit.json";
import judgedRules from "../tests/fixtures/judged-rules.json";
import unexpectedStop from "../tests/fixtures/unexpected-stop.json";

type Sample = { request: { state: unknown; questions: object } };

const SAMPLES: Record<string, Sample> = {
  "Auto thinking level (choice)": autoThinking,
  "Unexpected stop (noul with criteria)": unexpectedStop,
  "Judged rules (noul)": judgedRules,
  "Git AI staging (score)": gitStageUnit,
  "find: rank filenames (noul)": findNames,
  "find: score sketches (noul)": findSketches,
  "find: verify passages (noul)": findPassages,
  "eval judge(), text state": evalTextState,
  "eval judge(), array state": evalArrayState,
  Custom: { request: { state: "", questions: { q: { type: "noul", instructions: "" } } } },
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const sample = $<HTMLSelectElement>("sample");
const state = $<HTMLTextAreaElement>("state");
const questions = $<HTMLTextAreaElement>("questions");
const status = $("status");
const answers = $("answers");
const raw = $("raw");

for (const name of Object.keys(SAMPLES)) sample.add(new Option(name));

function load() {
  const { state: s, questions: q } = SAMPLES[sample.value]!.request;
  state.value = typeof s === "string" ? s : JSON.stringify(s, null, 2);
  questions.value = JSON.stringify(q, null, 2);
}
sample.onchange = load;
load();

// omp sends `state` as text, a JSON object, or a JSON array. A JSON object or
// array in the box goes as JSON; anything else goes as text.
function stateValue(text: string): unknown {
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed !== null && typeof parsed === "object") return parsed;
  } catch {}
  return text;
}

// The shape laya-shim returns from POST /v1/systemone.
type Answer = {
  type: string;
  confidence?: number;
  noul?: number;
  score?: number;
  choice?: string;
  legend?: Record<string, string>;
  probabilities?: Record<string, number>;
};
type Reply = {
  answers?: Record<string, Answer>;
  usage?: { input_tokens: number };
  error?: string;
};

function summary(a: Answer): string {
  if (a.noul !== undefined) return a.noul >= 0.5 ? `yes (${a.noul})` : `no (${a.noul})`;
  if (a.score !== undefined) {
    const nearest = a.legend?.[Math.round(a.score)];
    return nearest ? `${a.score} (≈ ${nearest})` : String(a.score);
  }
  return a.choice ?? "";
}

function probabilities(a: Answer): string {
  return Object.entries(a.probabilities ?? {})
    .map(([k, p]) => `${a.legend?.[k] ?? k}: ${p}`)
    .join(", ");
}

function escape(s: string) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

$("ask").onclick = async () => {
  answers.innerHTML = "";
  raw.textContent = "";
  let parsed;
  try {
    parsed = JSON.parse(questions.value);
  } catch (e) {
    status.innerHTML = `<span class="error">Questions aren't valid JSON: ${escape(String(e))}</span>`;
    return;
  }
  status.textContent = "Asking…";
  const started = performance.now();
  const res = await fetch("/api/systemone", {
    method: "POST",
    body: JSON.stringify({ model: "laya-local", state: stateValue(state.value), questions: parsed }),
  });
  // Unchecked: this page only talks to the local shim, and it shows the raw reply too.
  const body = (await res.json()) as Reply;
  const ms = (performance.now() - started).toFixed(0);
  raw.textContent = JSON.stringify(body, null, 2);
  if (!res.ok) {
    status.innerHTML = `<span class="error">HTTP ${res.status}: ${escape(body.error ?? "")}</span>`;
    return;
  }
  status.textContent = `${ms} ms, ${body.usage?.input_tokens ?? "?"} input tokens`;
  const rows = Object.entries(body.answers ?? {})
    .map(
      ([id, a]) =>
        `<tr><td>${escape(id)}</td><td>${escape(a.type)}</td><td>${escape(summary(a))}</td>` +
        `<td>${a.confidence ?? ""}</td><td>${escape(probabilities(a))}</td></tr>`,
    )
    .join("");
  answers.innerHTML =
    "<table><tr><th>Question</th><th>Type</th><th>Answer</th><th>Confidence</th>" +
    `<th>Probabilities</th></tr>${rows}</table>`;
};
