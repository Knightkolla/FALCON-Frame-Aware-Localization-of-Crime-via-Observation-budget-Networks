"""
server/main.py — FALCON FastAPI Environment Server
===================================================
OpenEnv-compliant environment exposing 3 required endpoints + health.

Endpoints:
  POST /reset?task_id={1,2,3}  → Observation
  POST /step                   → StepResponse
  GET  /state                  → EpisodeState
  GET  /health                 → {"status": "ok"}

Architecture:
  - This file wraps the FALCON inference loop in HTTP endpoints.
  - The CCTVCosineEnv (adapted from WSI_cosine_env.py) handles step logic.
  - The FGlobal MLP (TSU from modules/fglobal_mlp.py) handles state updates.
  - Episode state is stored in-memory (single active episode per server instance).
"""

import sys
import os
import re
from types import SimpleNamespace
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add falcon_core to path so we can import the proven RL/TSU modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "falcon_core"))

from modules.fglobal_mlp import FGlobal

from server.CCTV_cosine_env import CCTVCosineEnv
from server.data_loader import (
    get_scenario,
    get_segment_by_id,
    scenario_to_hr_tensor,
    scenario_to_lr_tensor,
    get_all_partial_hints,
    list_available_tasks,
)
from server.grader import compute_reward, compute_no_crime_reward
from server.models import (
    Action,
    EpisodeState,
    ExpandedFeatures,
    LowResFeatures,
    Observation,
    Reward,
    SegmentObservation,
    StepResponse,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FALCON Environment",
    description=(
        "Frame-Aware Localization of Crime via Observation-budget Networks. "
        "OpenEnv-compliant CCTV crime frame localization environment. "
        "Built by Dhavala Kartikeya Somayaji."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config (FALCON environment parameters)
# ---------------------------------------------------------------------------

DEVICE = "cpu"  # CPU-only — no GPU needed, satisfies vcpu=2 memory=8GB constraint

FALCON_CONF = SimpleNamespace(
    frac_visit=0.20,          # 20% budget — FALCON-0.2
    cosine_threshold=0.85,    # cosine similarity threshold for TSU
    D_feat=128,               # FALCON feature dimension
    D_inner=64,               # reduced inner dim for lighter model
    dim_reduction=True,
    device=DEVICE,
)

# ---------------------------------------------------------------------------
# Load TSU (FGlobal MLP) — Targeted State Updater
# ---------------------------------------------------------------------------

# ip_dim = 128 * 3 (v_at ‖ z_at ‖ state[j]), op_dim = 128
TSU = FGlobal(ip_dim=128 * 3, op_dim=128, hidden_dim=256).to(DEVICE)
TSU.eval()
# Note: In production, load a pre-trained TSU checkpoint here:
# TSU.load_state_dict(torch.load("checkpoints/tsu_best.pt", map_location=DEVICE)["model"])
# For the hackathon, random weights simulate the TSU update.

# ---------------------------------------------------------------------------
# Episode state (single-session, in-memory)
# ---------------------------------------------------------------------------

_episode: dict = {
    "active": False,
    "task_id": None,
    "scenario": None,
    "env": None,
    "current_state": None,   # (1, N, 128) tensor
    "visited_ids": [],        # list of segment ID strings
    "flagged_ids": [],        # list of flagged segment ID strings
    "budget_used": 0,
    "budget_total": 0,
    "ground_truth_frame": None,
    "done": False,
    "steps_taken": 0,
    "final_reward": None,
}


# ---------------------------------------------------------------------------
# Helper: build Observation from current episode state
# ---------------------------------------------------------------------------

def _build_observation(include_hidden: bool = False) -> Observation:
    """Convert internal episode state to the Observation Pydantic model."""
    scenario = _episode["scenario"]
    state_tensor = _episode["current_state"]

    segments = []
    for i, raw_seg in enumerate(scenario["segments"]):
        seg_id = raw_seg["id"]
        is_expanded = seg_id in _episode["visited_ids"]
        is_flagged = seg_id in _episode["flagged_ids"]

        lr = LowResFeatures(
            motion_score=raw_seg["low_res"]["motion_score"],
            brightness_change=raw_seg["low_res"]["brightness_change"],
            person_count=raw_seg["low_res"]["person_count"],
            lr_feature_vector=state_tensor[0][i].tolist(),  # may be updated by TSU
        )

        expanded = None
        if is_expanded:
            raw_exp = raw_seg["expanded"]
            expanded = ExpandedFeatures(
                person_trajectories=raw_exp["person_trajectories"],
                hr_feature_vector=raw_exp["hr_feature_vector"],
                partial_reward_hint=raw_exp["partial_reward_hint"],
            )

        segments.append(
            SegmentObservation(
                id=seg_id,
                start_frame=raw_seg["start_frame"],
                end_frame=raw_seg["end_frame"],
                low_res=lr,
                is_expanded=is_expanded,
                is_flagged=is_flagged,
                expanded=expanded,
            )
        )

    return Observation(
        task_id=scenario["task_id"],
        crime_type=scenario["crime_type"],
        difficulty=scenario["difficulty"],
        description=scenario["description"],
        total_segments=scenario["total_segments"],
        budget_remaining=_episode["budget_total"] - _episode["budget_used"],
        budget_total=_episode["budget_total"],
        episode_done=_episode["done"],
        flagged_segments=list(_episode["flagged_ids"]),
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Action parser — converts text string to structured action
# ---------------------------------------------------------------------------

def _parse_action(raw: str):
    """
    Parse action string into (action_type, segment_idx_or_none, frame_or_none).

    Valid formats:
      "expand segment_42"    → ("expand", 42, None)
      "flag segment_7"       → ("flag", 7, None)
      "submit_frame 1350"    → ("submit_frame", None, 1350)
      "submit_no_crime"      → ("submit_no_crime", None, None)
    """
    raw = raw.strip().lower()

    m = re.match(r"^expand\s+segment_(\d+)$", raw)
    if m:
        return "expand", int(m.group(1)), None

    m = re.match(r"^flag\s+segment_(\d+)$", raw)
    if m:
        return "flag", int(m.group(1)), None

    m = re.match(r"^submit_frame\s+(\d+)$", raw)
    if m:
        return "submit_frame", None, int(m.group(1))

    if raw == "submit_no_crime":
        return "submit_no_crime", None, None

    raise ValueError(
        f"Unrecognised action: '{raw}'. "
        "Valid: 'expand segment_N', 'flag segment_N', 'submit_frame N', 'submit_no_crime'"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check. Must return 200."""
    return {"status": "ok", "project": "FALCON", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks (convenience endpoint)."""
    return {"tasks": list_available_tasks()}


@app.post("/reset", response_model=Observation)
def reset(task_id: int = Query(default=1, ge=1, le=3, description="Task ID (1=Robbery, 2=Shoplifting, 3=Fighting)")):
    """
    Reset the environment and start a new episode.

    Initialises the episode:
    - Loads scenario for task_id
    - Initialises CCTVCosineEnv with lr_features + hr_features
    - Returns initial Observation (all segments at low-res, none expanded)
    """
    scenario = get_scenario(task_id)
    lr_tensor = scenario_to_lr_tensor(scenario, device=DEVICE)   # (1, N, 128)
    hr_tensor = scenario_to_hr_tensor(scenario, device=DEVICE)   # (N, 128)
    hints = get_all_partial_hints(scenario)

    env = CCTVCosineEnv(
        lr_features=lr_tensor,
        hr_features=hr_tensor,
        partial_hints=hints,
        ground_truth_frame=scenario["ground_truth_frame"],
        conf=FALCON_CONF,
    )
    env.reset()

    _episode.update({
        "active": True,
        "task_id": task_id,
        "scenario": scenario,
        "env": env,
        "current_state": lr_tensor.clone(),
        "visited_ids": [],
        "flagged_ids": [],
        "budget_used": 0,
        "budget_total": scenario["budget"],
        "ground_truth_frame": scenario["ground_truth_frame"],
        "done": False,
        "steps_taken": 0,
        "final_reward": None,
    })

    return _build_observation()


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """
    Execute one action in the environment.

    Execute one action in the FALCON environment.
      while not done:
          action = agent.get_action(state, visited)
          new_state, reward, done = env.step(action, fglobal, classifier, device)
          state = new_state

    For FALCON: the LLM agent calls this endpoint instead of the Python loop.
    """
    if not _episode["active"]:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    if _episode["done"]:
        raise HTTPException(status_code=400, detail="Episode already done. Call POST /reset to start a new one.")

    scenario = _episode["scenario"]
    env: CCTVCosineEnv = _episode["env"]

    try:
        action_type, seg_idx, frame_num = _parse_action(action.raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    reward: Optional[Reward] = None
    done = False
    info = {}

    # ------------------------------------------------------------------
    if action_type == "expand":
        # Validate segment index
        if seg_idx < 0 or seg_idx >= scenario["total_segments"]:
            raise HTTPException(status_code=422, detail=f"segment_{seg_idx} out of range [0, {scenario['total_segments']-1}]")

        seg_id = f"segment_{seg_idx}"
        if seg_id in _episode["visited_ids"]:
            raise HTTPException(status_code=422, detail=f"{seg_id} already expanded.")

        if _episode["budget_used"] >= _episode["budget_total"]:
            raise HTTPException(status_code=422, detail="Budget exhausted. You must submit now.")

        # Call CCTVCosineEnv.step()
        action_tensor = torch.tensor(seg_idx)
        new_state, step_reward, env_done = env.step(
            action=action_tensor,
            state_update_net=TSU,
            device=DEVICE,
        )

        _episode["current_state"] = new_state
        _episode["visited_ids"].append(seg_id)
        _episode["budget_used"] += 1
        _episode["steps_taken"] += 1

        info["step_reward_hint"] = step_reward
        info["segments_expanded"] = len(_episode["visited_ids"])
        info["budget_remaining"] = _episode["budget_total"] - _episode["budget_used"]

        # Auto-end if budget exhausted
        if _episode["budget_used"] >= _episode["budget_total"]:
            info["message"] = "Budget exhausted — submit your answer now."

    # ------------------------------------------------------------------
    elif action_type == "flag":
        seg_id = f"segment_{seg_idx}"
        if seg_id not in _episode["flagged_ids"]:
            _episode["flagged_ids"].append(seg_id)
        info["message"] = f"{seg_id} flagged as suspicious. No budget cost."
        _episode["steps_taken"] += 1

    # ------------------------------------------------------------------
    elif action_type == "submit_frame":
        reward = compute_reward(
            submitted_frame=frame_num,
            ground_truth_frame=_episode["ground_truth_frame"],
            budget_used=_episode["budget_used"],
            budget_total=_episode["budget_total"],
        )
        _episode["done"] = True
        _episode["final_reward"] = reward
        _episode["steps_taken"] += 1
        done = True
        env.reset()  # clean up env state
        info["message"] = f"Episode complete. Score: {reward.score}"

    # ------------------------------------------------------------------
    elif action_type == "submit_no_crime":
        reward = compute_no_crime_reward(
            budget_used=_episode["budget_used"],
            budget_total=_episode["budget_total"],
            crime_actually_present=True,  # all 3 tasks have crimes
        )
        _episode["done"] = True
        _episode["final_reward"] = reward
        _episode["steps_taken"] += 1
        done = True
        env.reset()
        info["message"] = f"Episode complete. Score: {reward.score}"

    return StepResponse(
        observation=_build_observation(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EpisodeState)
def state():
    """Return current episode state summary (for debugging / agent inspection)."""
    if not _episode["active"]:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")

    return EpisodeState(
        task_id=_episode["task_id"],
        budget_used=_episode["budget_used"],
        budget_remaining=_episode["budget_total"] - _episode["budget_used"],
        visited_segments=list(_episode["visited_ids"]),
        flagged_segments=list(_episode["flagged_ids"]),
        episode_done=_episode["done"],
        steps_taken=_episode["steps_taken"],
    )


# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=True)
