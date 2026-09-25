"""Status codes omp acts on, checked without the checkpoint.

omp's TypeSafe client retries 408, 429, and 5xx, and it moves on to the next
judge after any other error. A bad request must come back as 422, so omp
doesn't wait on retries of a request that can never succeed.
"""

import pytest
from conftest import Shim

REQUEST = {
    "model": "laya-local",
    "state": "All tests pass now.",
    "questions": {"q": {"type": "noul", "instructions": "Does it claim tests pass?"}},
}


class Agent:
    def __init__(self, error=None):
        self.error = error

    def predict(self, state, questions):
        if self.error:
            raise self.error
        return {
            "model": "stub",
            "answers": {"q": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


@pytest.fixture
def shim(request):
    shim = Shim(Agent(getattr(request, "param", None)))
    yield shim
    shim.close()


def test_answers_with_200(shim):
    status, body = shim.post(REQUEST)
    assert status == 200
    assert body["answers"]["q"]["noul"] == 0.9


def test_unknown_path_is_404(shim):
    status, _ = shim.post(REQUEST, path="/v1/chat/completions")
    assert status == 404


def test_invalid_json_is_422(shim):
    status, _ = shim.post(b"{not json")
    assert status == 422


@pytest.mark.parametrize("field", ["state", "questions"])
def test_missing_field_is_422(shim, field):
    body = {k: v for k, v in REQUEST.items() if k != field}
    status, reply = shim.post(body)
    assert status == 422
    assert field in reply["error"]


@pytest.mark.parametrize("shim", [ValueError("too many options")], indirect=True)
def test_question_laya_rejects_is_422(shim):
    status, body = shim.post(REQUEST)
    assert status == 422
    assert body["error"] == "too many options"


@pytest.mark.parametrize("shim", [RuntimeError("boom")], indirect=True)
def test_inference_crash_is_500(shim):
    status, body = shim.post(REQUEST)
    assert status == 500
    assert body["error"] == "inference failed"
