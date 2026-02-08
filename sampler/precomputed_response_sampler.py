"""
Sampler that returns precomputed response_text from a jsonl (e.g. ARK-generated).

Used for Phase 2: grade HealthBench with responses produced offline (e.g. by
scripts/run_ark_on_healthbench.py). Responses must be in the same order as the eval examples.
Each __call__(message_list) returns the next line's response_text; the message_list
is ignored (we only use the precomputed text). HealthBenchEval calls the sampler
once per example, so the jsonl must have exactly that many lines in the same order.
"""

import json

from ..types import MessageList, SamplerBase, SamplerResponse


class PrecomputedResponseSampler(SamplerBase):
    """
    Return precomputed responses in order. No model is called. Used so we can
    grade ARK (or other) responses with the same HealthBench eval code.
    """

    def __init__(
        self,
        response_jsonl_path: str | None = None,
        responses: list[dict] | None = None,
    ):
        """
        Either response_jsonl_path or responses must be provided.
        - response_jsonl_path: path to jsonl with one line per example; each line has "response_text"
          (and optionally "prompt", "rubrics", "example_tags").
        - responses: list of dicts with "response_text" key (same order as eval examples).
        """
        if response_jsonl_path is not None:
            self._responses = []
            with open(response_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self._responses.append(json.loads(line))
        elif responses is not None:
            self._responses = list(responses)
        else:
            raise ValueError("Provide response_jsonl_path or responses")

        # HealthBenchEval calls __call__ once per example in order; we serve the next line each time.
        self._index = 0

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """
        Return the next precomputed response. message_list is the prompt HealthBench passed
        (we don't use it to generate; we only return the next stored response_text).
        actual_queried_message_list is set to message_list so the eval can log the convo.
        """
        if self._index >= len(self._responses):
            raise IndexError(
                f"PrecomputedResponseSampler: no response for index {self._index} (have {len(self._responses)} responses)"
            )
        row = self._responses[self._index]
        self._index += 1
        response_text = row.get("response_text", "")
        return SamplerResponse(
            response_text=response_text,
            actual_queried_message_list=message_list,
            response_metadata={},
        )

