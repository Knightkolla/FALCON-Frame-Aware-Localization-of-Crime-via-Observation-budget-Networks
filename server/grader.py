"""
server/grader.py — FALCON Frame Proximity Reward
=================================================
Deterministic reward function — no LLM needed.
Mirrors FALCON's reward (negative cross-entropy → 0-1 score),
adapted for frame-level regression instead of classification.

Scoring table:
  exact frame           → 1.00
  within  1s (30f)      → 0.90  + efficiency bonus
  within  3s (90f)      → 0.70  + efficiency bonus
  within 10s (300f)     → 0.40  + efficiency bonus
  within 30s (900f)     → 0.10
  beyond 30s            → 0.00
  submit_no_crime (wrong) → 0.00
"""

from server.models import Reward

FPS = 30  # CCTV frame rate


def compute_reward(
    submitted_frame: int,
    ground_truth_frame: int,
    budget_used: int,
    budget_total: int,
    crime_present: bool = True,
) -> Reward:
    """
    Compute the proximity-based reward.

    Args:
        submitted_frame:     Frame number the agent claims crime starts.
        ground_truth_frame:  Actual crime start frame.
        budget_used:         How many expand actions the agent used.
        budget_total:        Maximum allowed expand actions (20% of N).
        crime_present:       True for tasks 1-3; False if agent submits submit_no_crime.

    Returns:
        Reward model with score in [0, 1].
    """
    distance = abs(submitted_frame - ground_truth_frame)

    # Base score from proximity tiers
    if distance == 0:
        base_score = 1.00
    elif distance <= FPS * 1:    # within 1 second
        base_score = 0.90
    elif distance <= FPS * 3:    # within 3 seconds
        base_score = 0.70
    elif distance <= FPS * 10:   # within 10 seconds
        base_score = 0.40
    elif distance <= FPS * 30:   # within 30 seconds
        base_score = 0.10
    else:
        base_score = 0.00

    # Efficiency bonus — reward using fewer budget units (like FALCON's speed bonus)
    # Only applied when base_score > 0 (no bonus for wrong answers)
    if base_score > 0 and budget_total > 0:
        efficiency_ratio = 1.0 - (budget_used / budget_total)
        efficiency_bonus = round(0.10 * efficiency_ratio * base_score, 4)
    else:
        efficiency_bonus = 0.0

    final_score = min(0.99, max(0.01, round(base_score + efficiency_bonus, 4)))

    # Human-readable explanation
    seconds_off = distance / FPS
    if distance == 0:
        detail = "Exact match! Perfect score."
    elif distance <= FPS * 30:
        detail = f"Off by {distance} frames ({seconds_off:.1f}s)."
    else:
        detail = f"Off by {distance} frames ({seconds_off:.1f}s) — beyond 30s window, no score."

    budget_pct = round(100 * budget_used / budget_total, 1) if budget_total > 0 else 0
    explanation = (
        f"{detail} "
        f"Base score: {base_score:.2f}. "
        f"Efficiency bonus: +{efficiency_bonus:.4f} "
        f"(used {budget_used}/{budget_total} = {budget_pct}% of budget). "
        f"Final: {final_score:.4f}."
    )

    return Reward(
        score=final_score,
        submitted_frame=submitted_frame,
        ground_truth_frame=ground_truth_frame,
        frame_distance=distance,
        efficiency_bonus=efficiency_bonus,
        explanation=explanation,
    )


def compute_no_crime_reward(budget_used: int, budget_total: int, crime_actually_present: bool) -> Reward:
    """
    Reward for submit_no_crime action.
    Tasks 1-3 always have a crime — submitting no_crime is always wrong here.
    """
    if crime_actually_present:
        return Reward(
            score=0.01,
            submitted_frame=None,
            ground_truth_frame=None,
            frame_distance=None,
            efficiency_bonus=0.0,
            explanation="Agent submitted 'no crime' but a crime was present. Score: 0.0.",
        )
    else:
        # Correct no-crime decision (not used in current tasks but future-proof)
        return Reward(
            score=0.99,
            submitted_frame=None,
            ground_truth_frame=None,
            frame_distance=None,
            efficiency_bonus=0.0,
            explanation="Correctly identified clean video. Score: 1.0.",
        )
