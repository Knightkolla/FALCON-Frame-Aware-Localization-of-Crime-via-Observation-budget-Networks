"""
inference.py — FALCON LLM Agent
================================
OpenEnv baseline inference script. Must be named exactly `inference.py` at root.

Runs all 3 tasks using an LLM (OpenAI-compatible API) as the agent.
The LLM reads the observation JSON and decides which action to take,
mirroring how a human analyst would scan CCTV footage.

Usage:
  OPENAI_API_KEY=sk-... python inference.py

  # Against HF Space:
  FALCON_ENV_URL=https://your-space.hf.space python inference.py

  # Against local server:
  FALCON_ENV_URL=http://localhost:8000 python inference.py
"""

import json
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("FALCON_ENV_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("FALCON_LLM_MODEL", "gpt-4o-mini")
MAX_STEPS_PER_TASK = 200   # safety cap

# ---------------------------------------------------------------------------
# OpenAI client (compatible with any OpenAI-spec endpoint)
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    USE_LLM = bool(OPENAI_API_KEY)
except ImportError:
    USE_LLM = False
    print("WARNING: openai package not found. Running heuristic baseline instead.")


# ---------------------------------------------------------------------------
# System prompt — describes the FALCON task to the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are FALCON, an AI agent for CCTV crime frame localization.

You observe a CCTV video represented as N temporal segments, each with:
- motion_score (0-1): how much movement in this 1-second window
- brightness_change (0-1): sudden lighting change
- person_count: number of people visible
- is_expanded: whether you've already seen full 30fps detail
- expanded.person_trajectories: movement description (only after expanding)
- expanded.partial_reward_hint: how close this segment is to the crime (0=far, 1=crime here)

Your goal: find the EXACT FRAME where a crime starts. You have a limited budget of expansions.

Available actions (respond with ONLY the action string, nothing else):
  expand segment_{N}     — spend 1 budget unit to see full detail of segment N
  flag segment_{N}       — mark segment as suspicious (free, no budget cost)
  submit_frame {F}       — submit frame number F as your answer (ends episode)
  submit_no_crime        — declare video is clean (only if NO crime exists)

Strategy:
1. Scan the low-res features for anomalies (high motion_score + brightness_change)
2. Expand the most suspicious segments first
3. Use partial_reward_hint in expanded segments to guide further exploration
4. When you find the crime segment, calculate the exact frame:
   frame = segment_start_frame + offset_within_segment
5. Submit before budget runs out

Current task information will be provided in each message.
Respond with ONLY a single valid action string."""


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def api_reset(task_id: int) -> dict:
    """POST /reset and return the observation (without segments for display)."""
    r = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def api_step(action_str: str) -> dict:
    """POST /step with an action string."""
    r = requests.post(
        f"{BASE_URL}/step",
        json={"raw": action_str},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_health() -> bool:
    """Check server is up."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Observation summariser (keeps LLM context small)
# ---------------------------------------------------------------------------

def summarise_observation(obs: dict) -> str:
    """
    Compact text summary of the current observation for the LLM.
    Avoids sending full 128-dim vectors in the prompt.
    """
    lines = [
        f"Task {obs['task_id']}: {obs['crime_type'].upper()} ({obs['difficulty']})",
        f"Description: {obs['description']}",
        f"Budget: {obs['budget_remaining']}/{obs['budget_total']} expansions remaining",
        f"Segments expanded: {sum(1 for s in obs['segments'] if s['is_expanded'])} / {obs['total_segments']}",
        "",
        "TOP 10 MOST SUSPICIOUS SEGMENTS (by motion_score, not yet expanded):",
    ]

    # Sort unexpanded segments by motion_score descending
    unexpanded = [s for s in obs["segments"] if not s["is_expanded"]]
    top = sorted(unexpanded, key=lambda s: s["low_res"]["motion_score"], reverse=True)[:10]
    for s in top:
        lines.append(
            f"  {s['id']} | frames {s['start_frame']}-{s['end_frame']} "
            f"| motion={s['low_res']['motion_score']:.3f} "
            f"| brightness={s['low_res']['brightness_change']:.3f} "
            f"| persons={s['low_res']['person_count']}"
            + (" [FLAGGED]" if s["is_flagged"] else "")
        )

    # Show already-expanded segments with their hints
    expanded = [s for s in obs["segments"] if s["is_expanded"] and s.get("expanded")]
    if expanded:
        lines.append("")
        lines.append("EXPANDED SEGMENTS (with crime proximity hints):")
        for s in sorted(expanded, key=lambda s: s["expanded"]["partial_reward_hint"], reverse=True):
            lines.append(
                f"  {s['id']} | frames {s['start_frame']}-{s['end_frame']} "
                f"| hint={s['expanded']['partial_reward_hint']:.3f} "
                f"| trajectory: {s['expanded']['person_trajectories']}"
            )

    if obs["flagged_segments"]:
        lines.append(f"\nFlagged segments: {', '.join(obs['flagged_segments'])}")

    lines.append(f"\nEpisode done: {obs['episode_done']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM agent decision
# ---------------------------------------------------------------------------

def llm_decide(obs_summary: str, conversation_history: list) -> str:
    """Get the next action from the LLM."""
    conversation_history.append({"role": "user", "content": obs_summary})
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        temperature=0.1,
        max_tokens=50,
    )
    action = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": action})
    return action


# ---------------------------------------------------------------------------
# Heuristic baseline (no LLM — for testing without API key)
# ---------------------------------------------------------------------------

def heuristic_decide(obs: dict) -> str:
    """
    Smart 3-phase heuristic agent:

    Phase 1 (Coarse scan): Expand evenly-spaced probe segments across the timeline
            to get partial_reward_hints covering the full video.
    Phase 2 (Binary search): Use hints to zoom into the hottest region,
            expanding the midpoint between the two highest-hint neighbours.
    Phase 3 (Submit): When a segment with hint >= 0.85 is found, or budget
            is nearly exhausted, submit the best guess.

    This works for ALL task types — it does NOT rely on motion_score, which
    is deliberately misleading in Task 2 (shoplifting).
    """
    all_segs = obs["segments"]
    n_total = obs["total_segments"]
    expanded = [s for s in all_segs if s["is_expanded"] and s.get("expanded")]
    unexpanded = [s for s in all_segs if not s["is_expanded"]]
    budget_left = obs["budget_remaining"]

    # --- Phase 3: Submit if we found a hot segment (hint >= 0.85) ---
    hot = [s for s in expanded if s["expanded"]["partial_reward_hint"] >= 0.85]
    if hot:
        best = max(hot, key=lambda s: s["expanded"]["partial_reward_hint"])
        return f"submit_frame {best['start_frame']}"

    # --- Submit if budget nearly exhausted ---
    if budget_left <= 2:
        if expanded:
            best = max(expanded, key=lambda s: s["expanded"]["partial_reward_hint"])
            return f"submit_frame {best['start_frame']}"
        return "submit_no_crime"

    if budget_left == 0:
        if expanded:
            best = max(expanded, key=lambda s: s["expanded"]["partial_reward_hint"])
            return f"submit_frame {best['start_frame']}"
        return "submit_no_crime"

    # --- Phase 1: Coarse scan — spread probes evenly across timeline ---
    # Use first ~30% of budget for evenly-spaced probes
    budget_total = obs["budget_total"]
    budget_used = budget_total - budget_left
    probe_budget = max(8, int(budget_total * 0.30))

    if budget_used < probe_budget:
        # Pick next evenly-spaced unexpanded segment
        step_size = max(1, n_total // probe_budget)
        for i in range(0, n_total, step_size):
            seg = all_segs[i]
            if not seg["is_expanded"]:
                return f"expand {seg['id']}"
        # If all probe spots taken, fall through to phase 2

    # --- Phase 2: Binary search toward highest hint ---
    if expanded:
        # Sort expanded by hint descending
        by_hint = sorted(expanded, key=lambda s: s["expanded"]["partial_reward_hint"], reverse=True)
        best_seg = by_hint[0]
        best_idx = int(best_seg["id"].split("_")[1])
        best_hint = best_seg["expanded"]["partial_reward_hint"]

        # If best hint is warm (>= 0.4), search neighbours for the peak
        if best_hint >= 0.4:
            # Try expanding adjacent unexpanded segments around the best
            for offset in [1, -1, 2, -2, 3, -3, 5, -5, 8, -8]:
                target_idx = best_idx + offset
                if 0 <= target_idx < n_total:
                    target_seg = all_segs[target_idx]
                    if not target_seg["is_expanded"]:
                        return f"expand {target_seg['id']}"

        # If best hint is lukewarm (0.1-0.4), try midpoint between top 2
        if len(by_hint) >= 2 and best_hint >= 0.1:
            second_idx = int(by_hint[1]["id"].split("_")[1])
            mid_idx = (best_idx + second_idx) // 2
            # Search around midpoint
            for offset in [0, 1, -1, 2, -2]:
                target_idx = mid_idx + offset
                if 0 <= target_idx < n_total:
                    target_seg = all_segs[target_idx]
                    if not target_seg["is_expanded"]:
                        return f"expand {target_seg['id']}"

    # --- Fallback: expand a random unexpanded segment ---
    if unexpanded:
        # Pick the middle unexpanded segment to maximize information
        mid = unexpanded[len(unexpanded) // 2]
        return f"expand {mid['id']}"

    # No segments left to expand, submit best guess
    if expanded:
        best = max(expanded, key=lambda s: s["expanded"]["partial_reward_hint"])
        return f"submit_frame {best['start_frame']}"
    return "submit_no_crime"


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: int) -> dict:
    """Run a full episode for task_id. Returns result dict with score."""
    print(f"\n{'='*60}")
    print(f"Task  TASK {task_id} — Starting episode")
    print(f"{'='*60}")

    obs = api_reset(task_id)
    print(f"  Crime type : {obs['crime_type']} ({obs['difficulty']})")
    print(f"  Segments   : {obs['total_segments']}")
    print(f"  Budget     : {obs['budget_total']} expansions (20%)")

    conversation_history = []
    steps = 0
    final_score = None
    final_reward = None

    while not obs["episode_done"] and steps < MAX_STEPS_PER_TASK:
        obs_summary = summarise_observation(obs)

        # Get action from LLM or heuristic
        if USE_LLM:
            try:
                action_str = llm_decide(obs_summary, conversation_history)
            except Exception as e:
                print(f"  WARNING: LLM error: {e}. Falling back to heuristic.")
                action_str = heuristic_decide(obs)
        else:
            action_str = heuristic_decide(obs)

        print(f"  Step {steps+1:03d} | Action: {action_str}")

        result = api_step(action_str)
        obs = result["observation"]
        steps += 1

        if result["done"]:
            final_reward = result["reward"]
            final_score = final_reward["score"]
            print(f"\n  Episode complete after {steps} steps")
            print(f"     Score       : {final_score}")
            print(f"     Explanation : {final_reward['explanation']}")
            break

        # Small delay to avoid hammering the server
        time.sleep(0.05)

    if final_score is None:
        print(f"  WARNING: Max steps reached without submission. Forcing submit.")
        expanded = [s for s in obs["segments"] if s["is_expanded"] and s.get("expanded")]
        if expanded:
            best = max(expanded, key=lambda s: s["expanded"]["partial_reward_hint"])
            result = api_step(f"submit_frame {best['start_frame']}")
            final_reward = result["reward"]
            final_score = final_reward["score"] if final_reward else 0.0
        else:
            final_score = 0.0

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": steps,
        "reward": final_reward,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("FALCON — LLM Crime Frame Localization Agent")
    print(f"   Environment : {BASE_URL}")
    print(f"   Agent       : {'LLM (' + MODEL + ')' if USE_LLM else 'Heuristic baseline'}")

    # Health check
    if not api_health():
        print(f"\nERROR: Cannot reach environment at {BASE_URL}")
        print("    Start the server with: uvicorn server.main:app --port 8000")
        sys.exit(1)

    print("   Server      : ✅ online\n")

    results = []
    for task_id in [1, 2, 3]:
        result = run_task(task_id)
        results.append(result)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        crime = {1: "Robbery", 2: "Shoplifting", 3: "Fighting"}[r["task_id"]]
        print(f"  Task {r['task_id']} ({crime:<12}):  {r['score']:.4f}  ({r['steps']} steps)")
        total += r["score"]
    avg = total / len(results)
    print(f"{'─'*60}")
    print(f"  Average score    :  {avg:.4f}")
    print(f"{'='*60}\n")

    # Write scores to file (for logging)
    with open("inference_scores.json", "w") as f:
        json.dump({"results": results, "average": avg}, f, indent=2)
    print("  Scores written to inference_scores.json")


if __name__ == "__main__":
    main()
