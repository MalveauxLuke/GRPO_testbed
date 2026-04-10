#!/usr/bin/env python3

from __future__ import annotations

import re
from typing import Any


def _normalize_ground_truth_int(ground_truth: str) -> int:
    ground_truth_text = str(ground_truth).strip()
    matches = re.findall(r"\b(\d{1,3})\b", ground_truth_text)
    if not matches:
        raise ValueError(f"Could not parse AIME integer ground truth from {ground_truth!r}")
    answer_int = int(matches[-1])
    if not 0 <= answer_int <= 999:
        raise ValueError(f"AIME integer ground truth is outside 0-999: {ground_truth!r}")
    return answer_int


def _stringify_parsed_answer(parsed_answer: Any) -> str:
    if isinstance(parsed_answer, list):
        return ", ".join(str(item) for item in parsed_answer)
    return str(parsed_answer)


def _is_empty_parse(parsed_answer: Any) -> bool:
    if parsed_answer is None:
        return True
    if isinstance(parsed_answer, (list, tuple, set, dict)) and len(parsed_answer) == 0:
        return True
    return False


def check_answer(model_output: str, ground_truth: str) -> dict[str, Any]:
    """
    Extract and verify an AIME answer from model output against ground truth.

    Returns:
        dict with:
            - extracted_answer: str or None
            - is_correct: bool or None
            - extraction_method: str
    """
    if model_output is None:
        return {
            "extracted_answer": None,
            "is_correct": None,
            "extraction_method": "extraction_failed",
        }

    model_output = str(model_output)

    # Step 1: Try math-verify's built-in extraction and verification.
    try:
        from math_verify import parse, verify

        parsed_answer = parse(model_output)
        gold = parse(str(ground_truth))
        if _is_empty_parse(parsed_answer) or _is_empty_parse(gold):
            raise ValueError("math-verify returned an empty parse.")
        is_correct = verify(gold, parsed_answer)
        return {
            "extracted_answer": _stringify_parsed_answer(parsed_answer),
            "is_correct": bool(is_correct),
            "extraction_method": "math_verify",
        }
    except Exception:
        pass

    # Step 2: Fallback — manual \boxed{} extraction.
    # AIME answers are always integers 0-999, so this is reliable.
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, model_output)

    if matches:
        raw_answer = matches[-1].strip()
        cleaned = re.sub(r"[\\$\s,]", "", raw_answer)
        int_matches = re.findall(r"\b(\d{1,3})\b", cleaned)
        if int_matches:
            extracted_int = int(int_matches[-1])
            gt_int = _normalize_ground_truth_int(ground_truth)
            return {
                "extracted_answer": str(extracted_int),
                "is_correct": extracted_int == gt_int,
                "extraction_method": "boxed_regex",
            }

        return {
            "extracted_answer": raw_answer,
            "is_correct": None,
            "extraction_method": "boxed_regex_unparsed",
        }

    # Step 3: Last resort — find the last integer in the final part of the output.
    int_matches = re.findall(r"\b(\d{1,3})\b", model_output[-500:])
    if int_matches:
        extracted_int = int(int_matches[-1])
        gt_int = _normalize_ground_truth_int(ground_truth)
        return {
            "extracted_answer": str(extracted_int),
            "is_correct": extracted_int == gt_int,
            "extraction_method": "last_integer_fallback",
        }

    return {
        "extracted_answer": None,
        "is_correct": None,
        "extraction_method": "extraction_failed",
    }


def ground_truth_sanity(problem_id: str, ground_truth: str) -> dict[str, Any] | None:
    try:
        answer_int = _normalize_ground_truth_int(ground_truth)
    except ValueError as exc:
        return {
            "problem_id": problem_id,
            "ground_truth": str(ground_truth),
            "issue": str(exc),
        }

    if not 0 <= answer_int <= 999:
        return {
            "problem_id": problem_id,
            "ground_truth": str(ground_truth),
            "issue": "AIME ground truth is outside expected integer range 0-999.",
        }
    if str(answer_int) != str(ground_truth).strip():
        return {
            "problem_id": problem_id,
            "ground_truth": str(ground_truth),
            "normalized_ground_truth": str(answer_int),
            "issue": "Ground truth required normalization to parse as an AIME integer.",
        }
    return None
