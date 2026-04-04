# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from verl.trainer.ppo.core_algos import _compute_gdpo_saturation_diagnostics
from verl.trainer.ppo.ray_trainer import _compute_gdpo_saturation_metrics


def _build_reward_inputs(*patterns: str) -> tuple[np.ndarray, np.ndarray]:
    reward_values = []
    group_index = []
    for group_id, pattern in enumerate(patterns):
        reward_values.extend(float(char) for char in pattern)
        group_index.extend([group_id] * len(pattern))
    return np.asarray(reward_values, dtype=np.float32), np.asarray(group_index, dtype=np.int64)


@pytest.mark.parametrize(
    ("pattern", "group_fraction", "all_zero_fraction", "all_one_fraction", "event_count"),
    [
        ("0000", 1.0, 1.0, 0.0, 1),
        ("1111", 1.0, 0.0, 1.0, 1),
        ("0101", 0.0, 0.0, 0.0, 0),
        ("0011", 0.0, 0.0, 0.0, 0),
    ],
)
def test_gdpo_saturation_single_group_examples(
    pattern: str,
    group_fraction: float,
    all_zero_fraction: float,
    all_one_fraction: float,
    event_count: int,
):
    reward_values, group_index = _build_reward_inputs(pattern)

    summary, events = _compute_gdpo_saturation_diagnostics(
        reward_values=reward_values,
        index=group_index,
        reward_name="correct_reward",
        epsilon=1e-6,
    )

    assert summary["group_count"] == 1
    assert summary["group_fraction"] == group_fraction
    assert summary["all_zero_fraction"] == all_zero_fraction
    assert summary["all_one_fraction"] == all_one_fraction
    assert len(events) == event_count


def test_gdpo_saturation_distinguishes_0000_1111_0101_groups():
    reward_values, group_index = _build_reward_inputs("0000", "1111", "0101")

    summary, events = _compute_gdpo_saturation_diagnostics(
        reward_values=reward_values,
        index=group_index,
        reward_name="correct_reward",
        epsilon=1e-6,
    )

    assert summary["group_count"] == 3
    assert summary["saturated_group_count"] == 2
    assert summary["all_zero_group_count"] == 1
    assert summary["all_one_group_count"] == 1
    assert summary["group_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["all_zero_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["all_one_fraction"] == pytest.approx(1.0 / 3.0)
    assert [event["group_id"] for event in events] == ["0", "1"]
    assert events[0]["is_all_zero"] is True
    assert events[0]["is_all_one"] is False
    assert events[1]["is_all_zero"] is False
    assert events[1]["is_all_one"] is True


def test_gdpo_saturation_metric_projection_handles_cross_reward_union():
    correct_reward, group_index = _build_reward_inputs("0000", "1111", "0101")
    format_reward, _ = _build_reward_inputs("0101", "1111", "1111")

    correct_summary, correct_events = _compute_gdpo_saturation_diagnostics(
        reward_values=correct_reward,
        index=group_index,
        reward_name="correct_reward",
        epsilon=1e-6,
    )
    format_summary, format_events = _compute_gdpo_saturation_diagnostics(
        reward_values=format_reward,
        index=group_index,
        reward_name="format_reward",
        epsilon=1e-6,
    )

    saturated_group_ids = {event["group_id"] for event in correct_events + format_events}
    gdpo_saturation_info = {
        "per_reward": {
            "correct_reward": correct_summary,
            "format_reward": format_summary,
        },
        "any_reward_group_fraction": float(len(saturated_group_ids) / 3.0),
        "events": correct_events + format_events,
    }

    metrics = _compute_gdpo_saturation_metrics(gdpo_saturation_info)

    assert metrics["gdpo_saturation/correct_reward/group_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["gdpo_saturation/correct_reward/all_zero_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["gdpo_saturation/correct_reward/all_one_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["gdpo_saturation/format_reward/group_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["gdpo_saturation/format_reward/all_zero_fraction"] == 0.0
    assert metrics["gdpo_saturation/format_reward/all_one_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["gdpo_saturation/any_reward_group_fraction"] == 1.0
