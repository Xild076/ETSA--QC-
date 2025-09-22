"""Convenience helpers for running benchmarks with presets and tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .benchmark import run_benchmark
from .presets import preset_details


def run_with_preset(
    run_name: str,
    dataset: str,
    *,
    preset: str = "full_stack",
    limit: Optional[int] = None,
    pos_thresh: float = 0.1,
    neg_thresh: float = -0.1,
) -> Dict[str, Any]:
    return run_benchmark(
        run_name=run_name,
        dataset_name=dataset,
        run_mode=preset,
        limit=limit,
        pos_thresh=pos_thresh,
        neg_thresh=neg_thresh,
    )


def presets() -> Dict[str, Dict[str, Any]]:
    return preset_details()
