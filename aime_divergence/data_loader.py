#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class AIMEProblem:
    problem_id: str
    problem_text: str
    ground_truth_answer: str
    dataset: str
    source: str
    source_row_index: int

    def to_json(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_text": self.problem_text,
            "ground_truth_answer": self.ground_truth_answer,
            "dataset": self.dataset,
            "source": self.source,
            "source_row_index": self.source_row_index,
        }


DATASET_SPECS = {
    "aime24": [
        ("HuggingFaceH4/aime_2024", None),
        ("Maxwell-Jia/AIME_2024", None),
        ("AI-MO/aimo-validation-aime", "2024"),
    ],
    "aime25": [
        ("MathArena/aime_2025", None),
        ("opencompass/AIME2025", None),
    ],
}


def _first_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    preferred_splits = ("train", "test", "validation", "eval")
    for split in preferred_splits:
        if split in dataset:
            return dataset[split]
    first_key = next(iter(dataset.keys()))
    return dataset[first_key]


def _get_first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        if field in row and row[field] is not None:
            return row[field]
    raise KeyError(f"Could not find any of fields {fields} in row keys {sorted(row.keys())}")


def _normalize_answer(answer: Any) -> str:
    text = str(answer).strip()
    matches = re.findall(r"\b(\d{1,3})\b", text)
    if not matches:
        raise ValueError(f"Could not normalize AIME answer from {answer!r}")
    return str(int(matches[-1]))


def _looks_like_year(row: dict[str, Any], year: str) -> bool:
    haystack = " ".join(str(value) for value in row.values() if value is not None)
    return year in haystack


def _normalize_rows(dataset_name: str, split: Dataset, dataset_label: str, filter_year: str | None) -> list[AIMEProblem]:
    problems: list[AIMEProblem] = []
    for row_index, raw_row in enumerate(split):
        row = dict(raw_row)
        if filter_year is not None and not _looks_like_year(row, filter_year):
            continue

        problem_text = str(_get_first_present(row, ("problem", "Problem", "question", "Question", "prompt"))).strip()
        answer_text = _normalize_answer(
            _get_first_present(row, ("answer", "Answer", "ground_truth", "final_answer", "label"))
        )

        problems.append(
            AIMEProblem(
                problem_id=f"{dataset_label}_{len(problems) + 1:02d}",
                problem_text=problem_text,
                ground_truth_answer=answer_text,
                dataset=dataset_label,
                source=dataset_name,
                source_row_index=row_index,
            )
        )
    problems.sort(key=lambda item: item.problem_id)
    return problems


def _load_dataset_with_fallback(dataset_label: str, expected_count: int = 30) -> tuple[list[AIMEProblem], str]:
    errors: list[str] = []
    for dataset_name, filter_year in DATASET_SPECS[dataset_label]:
        try:
            dataset = load_dataset(dataset_name)
            split = _first_split(dataset)
            problems = _normalize_rows(dataset_name, split, dataset_label, filter_year)
            if len(problems) != expected_count:
                raise ValueError(
                    f"Loaded {len(problems)} normalized problems from {dataset_name}; expected {expected_count}."
                )
            return problems, dataset_name
        except Exception as exc:
            errors.append(f"{dataset_name}: {exc}")

    joined_errors = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"Could not load {dataset_label} from any configured source:\n{joined_errors}")


def load_aime_2024_2025(expected_per_year: int = 30) -> tuple[list[AIMEProblem], dict[str, str]]:
    aime24, source24 = _load_dataset_with_fallback("aime24", expected_count=expected_per_year)
    aime25, source25 = _load_dataset_with_fallback("aime25", expected_count=expected_per_year)
    sources = {"aime24": source24, "aime25": source25}
    return aime24 + aime25, sources


def print_dataset_confirmation(problems: list[AIMEProblem], sources: dict[str, str]) -> None:
    for dataset_label in ("aime24", "aime25"):
        count = sum(1 for problem in problems if problem.dataset == dataset_label)
        print(f"[aime-debug] loaded {count} problems for {dataset_label} from {sources.get(dataset_label)}")
    print(f"[aime-debug] total problems loaded: {len(problems)}")
