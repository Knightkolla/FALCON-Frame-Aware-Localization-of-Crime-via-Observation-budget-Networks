"""
server/models.py — FALCON Pydantic Models
==========================================
All OpenEnv-compliant typed models.
These are the exact types returned by /reset, /step, /state endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Segment-level types
# ---------------------------------------------------------------------------

class LowResFeatures(BaseModel):
    """Features visible without expanding (from 1fps thumbnail scan)."""
    motion_score: float = Field(..., ge=0.0, le=1.0, description="Normalised motion intensity 0-1")
    brightness_change: float = Field(..., ge=0.0, le=1.0, description="Brightness delta vs previous segment")
    person_count: int = Field(..., ge=0, description="Number of visible persons")
    lr_feature_vector: List[float] = Field(..., description="128-dim low-res feature vector")


class ExpandedFeatures(BaseModel):
    """Features revealed only after the agent expands a segment (costs 1 budget unit)."""
    person_trajectories: str = Field(..., description="Trajectory description string from full-fps analysis")
    hr_feature_vector: List[float] = Field(..., description="128-dim high-res feature vector")
    partial_reward_hint: float = Field(..., ge=0.0, le=1.0, description="Informativeness hint (0=useless, 1=crime here)")


class SegmentObservation(BaseModel):
    """A single temporal segment in the observation."""
    id: str = Field(..., description="Segment identifier, e.g. 'segment_42'")
    start_frame: int = Field(..., ge=0, description="First frame index of this segment")
    end_frame: int = Field(..., ge=0, description="Last frame index of this segment")
    low_res: LowResFeatures
    is_expanded: bool = Field(default=False, description="True if agent has already expanded this segment")
    is_flagged: bool = Field(default=False, description="True if agent has flagged this segment as suspicious")
    expanded: Optional[ExpandedFeatures] = Field(
        default=None,
        description="Populated only after agent expands this segment"
    )


# ---------------------------------------------------------------------------
# Episode-level Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observation returned by /reset or /step.
    Mirrors FALCON's state representation: (N_segments, feature_dim),
    but serialised as structured JSON for the OpenEnv API.
    """
    task_id: int = Field(..., description="1=Robbery, 2=Shoplifting, 3=Fighting")
    crime_type: str = Field(..., description="Human-readable crime category")
    difficulty: str = Field(..., description="easy | medium | hard")
    description: str = Field(..., description="Scenario description")
    total_segments: int = Field(..., description="N — total number of temporal segments")
    budget_remaining: int = Field(..., ge=0, description="Remaining expand budget (starts at 20% of total_segments)")
    budget_total: int = Field(..., description="Initial budget (20% of total_segments)")
    episode_done: bool = Field(default=False)
    flagged_segments: List[str] = Field(default_factory=list, description="Segment IDs flagged by agent")
    segments: List[SegmentObservation] = Field(..., description="All N temporal segments")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    A text action submitted by the agent.

    Valid formats:
      expand segment_{id}        — costs 1 budget unit, reveals ExpandedFeatures
      flag segment_{id}          — free, marks segment as suspicious
      submit_frame {frame_number} — ends episode, triggers grader
      submit_no_crime            — ends episode, declares video is clean
    """
    raw: str = Field(
        ...,
        description="Action string. Examples: 'expand segment_42', 'submit_frame 1350', 'flag segment_7'"
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Grader output — only returned when the episode ends (submit_frame or submit_no_crime).
    Score mirrors FALCON's slide-level reward, adapted for frame-level regression.
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Final score 0.0 to 1.0")
    submitted_frame: Optional[int] = Field(None, description="Frame the agent submitted")
    ground_truth_frame: Optional[int] = Field(None, description="Actual crime start frame")
    frame_distance: Optional[int] = Field(None, description="abs(submitted - ground_truth) in frames")
    efficiency_bonus: float = Field(0.0, description="Bonus for using fewer budget units")
    explanation: str = Field(..., description="Human-readable score breakdown")


# ---------------------------------------------------------------------------
# Step Response
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    """Full response from POST /step."""
    observation: Observation
    reward: Optional[Reward] = Field(None, description="Non-null only when episode_done=True")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict, description="Debug/meta info")


# ---------------------------------------------------------------------------
# State (for GET /state)
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """Internal episode state returned by GET /state."""
    task_id: int
    budget_used: int
    budget_remaining: int
    visited_segments: List[str]
    flagged_segments: List[str]
    episode_done: bool
    steps_taken: int
