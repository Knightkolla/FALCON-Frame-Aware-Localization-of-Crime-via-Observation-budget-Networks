"""
server/data_loader.py - FALCON Scenario Loader
===============================================
Loads scenarios.json (multi-variant pool) and provides random variant
selection so every /reset call gives the agent a fresh episode.
"""

import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch

SCENARIOS_PATH = os.path.join(os.path.dirname(__file__), "..", "scenarios.json")


@lru_cache(maxsize=1)
def _load_scenarios() -> dict:
    """Load and cache scenarios.json. Called once on server start."""
    path = os.path.abspath(SCENARIOS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"scenarios.json not found at {path}. "
            "Run `python preprocess.py` first to generate it."
        )
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_random_variant(task_id: int) -> dict:
    """
    Return a randomly selected variant for the given task_id.
    This is called on every /reset, giving each episode a unique scenario.
    The agent cannot memorize the answer - it must learn a real search strategy.
    """
    data = _load_scenarios()
    for task in data["tasks"]:
        if task["task_id"] == task_id:
            variants = task["variants"]
            chosen = random.choice(variants)
            # Attach top-level task metadata to the variant
            chosen = dict(chosen)  # shallow copy
            chosen["task_id"] = task["task_id"]
            chosen["crime_type"] = task["crime_type"]
            chosen["difficulty"] = task["difficulty"]
            chosen["description"] = task["description"]
            return chosen
    raise ValueError(f"task_id={task_id} not found. Valid IDs: 1, 2, 3.")


def get_segment_by_id(scenario: dict, segment_id: str) -> Optional[dict]:
    for seg in scenario["segments"]:
        if seg["id"] == segment_id:
            return seg
    return None


def scenario_to_lr_tensor(scenario: dict, device: str = "cpu") -> torch.Tensor:
    """(1, N, 128) low-res feature tensor — initial agent state."""
    vectors = [seg["low_res"]["lr_feature_vector"] for seg in scenario["segments"]]
    arr = np.array(vectors, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def scenario_to_hr_tensor(scenario: dict, device: str = "cpu") -> torch.Tensor:
    """(N, 128) high-res feature tensor — revealed when agent expands a segment."""
    vectors = [seg["expanded"]["hr_feature_vector"] for seg in scenario["segments"]]
    arr = np.array(vectors, dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def get_all_partial_hints(scenario: dict) -> List[float]:
    return [seg["expanded"]["partial_reward_hint"] for seg in scenario["segments"]]


def list_available_tasks() -> List[Dict]:
    """Return task summaries (no variant/segment data)."""
    data = _load_scenarios()
    summaries = []
    for task in data["tasks"]:
        n_variants = len(task["variants"])
        n_no_crime = sum(1 for v in task["variants"] if v["is_clean"])
        example = task["variants"][n_no_crime]  # first non-clean variant
        summaries.append({
            "task_id": task["task_id"],
            "crime_type": task["crime_type"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "total_segments": example["total_segments"],
            "budget": example["budget"],
            "n_variants": n_variants,
            "n_no_crime_variants": n_no_crime,
            "note": "GT frame and red herrings randomized each episode.",
        })
    return summaries
