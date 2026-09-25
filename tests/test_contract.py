"""Checks the shim's replies against the System One contract omp relies on.

Each fixture in tests/fixtures holds a request copied from the omp feature it
names in `source`. The checks follow omp's `packages/ai/src/judgment/types.ts`
and the reply checks in `TypeSafeJudge.judge()`. These tests load the real
checkpoint, so they carry the `model` marker.
"""

import json
from pathlib import Path

import pytest
from conftest import Shim

from laya_shim.server import load_agent

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def shim():
    shim = Shim(load_agent())
    yield shim
    shim.close()


def check_distribution(probabilities, keys):
    assert set(probabilities) == set(keys)
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1, abs=2e-3)


def check_answer(question, answer):
    assert answer["type"] == question["type"]
    if question["type"] == "noul":
        assert 0 <= answer["noul"] <= 1
    elif question["type"] == "choice":
        options = list(question["criteria"])
        probabilities = answer["probabilities"]
        check_distribution(probabilities, options)
        assert answer["choice"] == max(options, key=probabilities.get)
        assert 0 <= answer["confidence"] <= 1
    elif question["type"] == "score":
        levels = [str(i) for i in range(len(question["criteria"]))]
        probabilities = answer["probabilities"]
        check_distribution(probabilities, levels)
        expected = sum(int(level) * probabilities[level] for level in levels)
        assert answer["score"] == pytest.approx(expected, abs=2e-3)
        assert 0 <= answer["confidence"] <= 1
    else:
        pytest.fail(f"fixture has unknown question type {question['type']!r}")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
def test_reply_matches_contract(shim, path):
    request = json.loads(path.read_text())["request"]
    status, reply = shim.post(request)
    assert status == 200, reply

    assert isinstance(reply["model"], str)
    usage = reply["usage"]
    assert isinstance(usage["input_tokens"], int) and usage["input_tokens"] > 0
    assert isinstance(usage["output_tokens"], int) and usage["output_tokens"] >= 0

    for qid, question in request["questions"].items():
        assert qid in reply["answers"], f"no answer for question {qid!r}"
        check_answer(question, reply["answers"][qid])
