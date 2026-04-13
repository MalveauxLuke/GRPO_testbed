from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aime_divergence.run_logged import (
    LogprobsExtractionError,
    _correctness_status,
    _load_partial_problem_rows,
    approx_entropy,
    append_partial_problem_row,
    build_problem_token_payload,
    build_rollout_results_payload,
    extract_token_data,
    should_skip_problem,
    write_npz_atomic,
)


class FakeLogprob:
    def __init__(self, logprob: float, rank: int | None = None, decoded_token: str | None = None):
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = decoded_token


class FakeOutput:
    def __init__(self, token_ids: list[int], logprobs: list[dict[int, FakeLogprob]]):
        self.token_ids = token_ids
        self.logprobs = logprobs


class FakeProblem:
    def __init__(self) -> None:
        self.problem_id = "aime24_01"
        self.problem_text = "Example problem"
        self.ground_truth_answer = "42"
        self.dataset = "aime24"


def test_extract_token_data_handles_mapping_payload_and_zero_bases_rank() -> None:
    output = FakeOutput(
        token_ids=[11, 22],
        logprobs=[
            {
                11: FakeLogprob(np.log(0.7), rank=1),
                12: FakeLogprob(np.log(0.2), rank=2),
                13: FakeLogprob(np.log(0.1), rank=3),
            },
            {
                22: FakeLogprob(np.log(0.5), rank=1),
                21: FakeLogprob(np.log(0.3), rank=2),
                20: FakeLogprob(np.log(0.2), rank=3),
            },
        ],
    )

    token_data = extract_token_data(output, top_k=4)

    assert token_data["token_ids"].dtype == np.int32
    assert token_data["token_probs"].dtype == np.float32
    np.testing.assert_allclose(token_data["token_probs"], np.asarray([0.7, 0.5], dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(token_data["token_ranks"], np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_allclose(
        token_data["cumulative_logprobs"],
        np.asarray([np.log(0.7), np.log(0.7) + np.log(0.5)], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_array_equal(token_data["top_k_ids"][0, :3], np.asarray([11, 12, 13], dtype=np.int32))
    np.testing.assert_allclose(token_data["top_k_probs"][0, :3], np.asarray([0.7, 0.2, 0.1], dtype=np.float32), atol=1e-6)
    assert token_data["token_entropies"][0] > 0.0


def test_extract_token_data_pads_when_fewer_than_top_k_candidates_are_returned() -> None:
    output = FakeOutput(
        token_ids=[31],
        logprobs=[
            {
                31: FakeLogprob(np.log(0.8), rank=1),
                30: FakeLogprob(np.log(0.2), rank=2),
            }
        ],
    )

    token_data = extract_token_data(output, top_k=5)

    assert token_data["top_k_ids"].shape == (1, 5)
    np.testing.assert_array_equal(token_data["top_k_ids"][0], np.asarray([31, 30, -1, -1, -1], dtype=np.int32))
    np.testing.assert_allclose(token_data["top_k_probs"][0], np.asarray([0.8, 0.2, 0.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_extract_token_data_raises_when_sampled_token_is_missing() -> None:
    output = FakeOutput(
        token_ids=[44],
        logprobs=[{45: FakeLogprob(np.log(0.9), rank=1)}],
    )

    with pytest.raises(LogprobsExtractionError, match="Sampled token id 44"):
        extract_token_data(output, top_k=3)


def test_approx_entropy_matches_uniform_three_way_distribution() -> None:
    logprobs = np.log(np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64))
    assert approx_entropy(logprobs) == pytest.approx(np.log(3.0), rel=1e-6, abs=1e-6)


def test_correctness_status_uses_negative_one_for_unknown() -> None:
    assert _correctness_status(True) == 1
    assert _correctness_status(False) == 0
    assert _correctness_status(None) == -1


def test_problem_npz_round_trip_preserves_shapes_dtypes_and_padding(tmp_path: Path) -> None:
    problem = FakeProblem()
    rollouts = [
        {"is_correct": True, "extracted_answer": "42", "extraction_method": "math_verify"},
        {"is_correct": None, "extracted_answer": None, "extraction_method": "boxed_regex_unparsed"},
    ]
    token_data_per_rollout = [
        {
            "token_ids": np.asarray([11, 12], dtype=np.int32),
            "token_probs": np.asarray([0.7, 0.6], dtype=np.float32),
            "token_ranks": np.asarray([0, 1], dtype=np.int32),
            "token_entropies": np.asarray([0.5, 0.4], dtype=np.float32),
            "cumulative_logprobs": np.asarray([np.log(0.7), np.log(0.7) + np.log(0.6)], dtype=np.float32),
            "top_k_ids": np.asarray([[11, 13, -1], [12, 10, -1]], dtype=np.int32),
            "top_k_probs": np.asarray([[0.7, 0.3, 0.0], [0.6, 0.4, 0.0]], dtype=np.float32),
        },
        {
            "token_ids": np.asarray([21], dtype=np.int32),
            "token_probs": np.asarray([0.9], dtype=np.float32),
            "token_ranks": np.asarray([0], dtype=np.int32),
            "token_entropies": np.asarray([0.2], dtype=np.float32),
            "cumulative_logprobs": np.asarray([np.log(0.9)], dtype=np.float32),
            "top_k_ids": np.asarray([[21, -1, -1]], dtype=np.int32),
            "top_k_probs": np.asarray([[0.9, 0.0, 0.0]], dtype=np.float32),
        },
    ]

    payload = build_problem_token_payload(problem, rollouts, token_data_per_rollout, top_k=3)
    npz_path = tmp_path / "problem.npz"
    write_npz_atomic(npz_path, payload)

    with np.load(npz_path, allow_pickle=False) as data:
        assert data["correctness_status"].dtype == np.int8
        np.testing.assert_array_equal(data["correctness_status"], np.asarray([1, -1], dtype=np.int8))
        assert data["token_ids"].dtype == np.int32
        assert data["token_probs"].dtype == np.float32
        assert data["top_k_ids"].shape == (2, 2, 3)
        assert data["top_k_probs"].shape == (2, 2, 3)
        np.testing.assert_array_equal(data["num_tokens"], np.asarray([2, 1], dtype=np.int32))
        np.testing.assert_array_equal(data["token_ids"][1], np.asarray([21, -1], dtype=np.int32))
        np.testing.assert_allclose(data["token_probs"][1], np.asarray([0.9, 0.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_array_equal(data["top_k_ids"][1, 1], np.asarray([-1, -1, -1], dtype=np.int32))


def test_resume_skip_requires_both_npz_and_manifest_row(tmp_path: Path) -> None:
    partial_path = tmp_path / "_rollout_results.partial.jsonl"
    token_path = tmp_path / "token_data" / "aime24_01.npz"
    token_path.parent.mkdir(parents=True)
    token_path.write_bytes(b"npz")

    assert not should_skip_problem("aime24_01", token_path, {})

    row = {"problem_id": "aime24_01", "rollouts": [], "num_correct": 0, "num_incorrect": 0, "num_unknown": 0, "split_ratio": "0/0"}
    append_partial_problem_row(partial_path, row)
    rows = {loaded_row["problem_id"]: loaded_row for loaded_row in _load_partial_problem_rows(partial_path)}

    assert should_skip_problem("aime24_01", token_path, rows)


def test_build_rollout_results_payload_matches_accumulated_rows() -> None:
    metadata = {"model": "demo"}
    rows = [
        {
            "problem_id": "aime24_01",
            "rollouts": [
                {"rollout_idx": 0, "num_tokens": 10, "extraction_method": "math_verify"},
                {"rollout_idx": 1, "num_tokens": 12, "extraction_method": "boxed_regex"},
            ],
            "num_correct": 1,
            "num_incorrect": 1,
            "num_unknown": 0,
            "split_ratio": "1/1",
        }
    ]

    payload = build_rollout_results_payload(metadata, rows, [])

    assert payload["metadata"] == metadata
    assert payload["problems"] == rows
    assert payload["summary"]["total_problems"] == 1
    assert payload["summary"]["total_rollouts"] == 2
    assert payload["summary"]["usable_problems"] == 1
    assert payload["summary"]["split_distribution"] == {"1/1": 1}
    assert payload["summary"]["total_correct"] == 1
    assert payload["summary"]["total_incorrect"] == 1
