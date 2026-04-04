"""
server/data_loader.py — FALCON Scenario Loader
===============================================
Loads scenarios.json and serves scenario/segment data to the environment.
Analogous to FALCON's build_HDF5_feat_dataset_2 but for our JSON format.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Load scenarios.json (cached after first read)
# ---------------------------------------------------------------------------

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


def get_scenario(task_id: int) -> dict:
    """Return the raw scenario dict for a given task_id (1, 2, or 3)."""
    data = _load_scenarios()
    for task in data["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"task_id={task_id} not found. Valid IDs: 1, 2, 3.")


def get_segment_by_id(scenario: dict, segment_id: str) -> Optional[dict]:
    """Return a segment dict by its string ID (e.g. 'segment_42')."""
    for seg in scenario["segments"]:
        if seg["id"] == segment_id:
            return seg
    return None


def get_segment_by_index(scenario: dict, idx: int) -> dict:
    """Return a segment dict by its integer index."""
    return scenario["segments"][idx]


# ---------------------------------------------------------------------------
# Tensor conversion helpers (mirrors FALCON's HDF5 → tensor pipeline)
# ---------------------------------------------------------------------------

def scenario_to_lr_tensor(scenario: dict, device: str = "cpu") -> torch.Tensor:
    """
    Convert all low-res feature vectors to a (1, N, 128) tensor.
    This is the initial state fed to the PPO Agent — exactly like FALCON's lr_features.

    Shape: (batch=1, N_segments, feature_dim=128)
    """
    vectors = [seg["low_res"]["lr_feature_vector"] for seg in scenario["segments"]]
    arr = np.array(vectors, dtype=np.float32)           # (N, 128)
    tensor = torch.from_numpy(arr).unsqueeze(0)         # (1, N, 128)
    return tensor.to(device)


def scenario_to_hr_tensor(scenario: dict, device: str = "cpu") -> torch.Tensor:
    """
    Convert all high-res (expanded) feature vectors to a (N, 128) tensor.
    This is hr_features in FALCON — the ground-truth high-res representation
    that gets substituted in when the agent expands a segment.

    Shape: (N_segments, feature_dim=128)
    """
    vectors = [seg["expanded"]["hr_feature_vector"] for seg in scenario["segments"]]
    arr = np.array(vectors, dtype=np.float32)           # (N, 128)
    tensor = torch.from_numpy(arr)                      # (N, 128)
    return tensor.to(device)


def get_all_partial_hints(scenario: dict) -> List[float]:
    """Return list of partial_reward_hint values for all segments."""
    return [seg["expanded"]["partial_reward_hint"] for seg in scenario["segments"]]


def list_available_tasks() -> List[Dict]:
    """Return summary info for all tasks (no segments)."""
    data = _load_scenarios()
    summaries = []
    for task in data["tasks"]:
        summaries.append({
            "task_id": task["task_id"],
            "crime_type": task["crime_type"],
            "difficulty": task["difficulty"],
            "total_segments": task["total_segments"],
            "budget": task["budget"],
            "description": task["description"],
        })
    return summaries
