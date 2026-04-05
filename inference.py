"""
inference.py — FALCON Baseline LLM Agent
========================================
Strictly follows the OpenEnv Hackathon requirements for structured stdout logging.
"""

import json
import os
import sys
import time
from typing import List, Optional

import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")
# LLM Router URL
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Environment Server URL
ENV_BASE_URL = os.getenv("FALCON_ENV_URL", "http://localhost:8000")
BENCHMARK = "falcon_cctv"

try:
    from openai import OpenAI
    # The client connects to the Hugging Face Router
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    USE_LLM = bool(HF_TOKEN)
except ImportError:
    USE_LLM = False

# ---------------------------------------------------------------------------
# Structured Logging Helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# API Interactors
# ---------------------------------------------------------------------------

def api_reset(task_id: int) -> dict:
    url = f"{ENV_BASE_URL}/reset?task_id={task_id}"
    resp = requests.post(url, headers={"accept": "application/json"})
    resp.raise_for_status()
    return resp.json()

def api_step(action_str: str) -> dict:
    url = f"{ENV_BASE_URL}/step"
    payload = {"raw": action_str}
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    return resp.json()

# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------
def heuristic_decide(obs: dict) -> str:
    expanded_segments = [s for s in obs["segments"] if s["is_expanded"] and s.get("expanded")]
    
    # Check if we have strong signal to submit
    if expanded_segments:
        best = max(expanded_segments, key=lambda s: s["expanded"]["partial_reward_hint"])
        if best["expanded"]["partial_reward_hint"] > 0.85:
            # Random offset within 30fps = 15
            frame = best["start_frame"] + 15
            return f"submit_frame {frame}"
            
    # Budget check
    remaining = obs["budget_remaining"]
    if remaining <= 1:
        if expanded_segments:
            best = max(expanded_segments, key=lambda s: s["expanded"]["partial_reward_hint"])
            return f"submit_frame {best['start_frame'] + 15}"
        else:
            return "submit_no_crime"

    # Not expanded yet
    unexpanded = [s for s in obs["segments"] if not s["is_expanded"]]
    if not unexpanded:
        return "submit_no_crime"
        
    # Pick based on highest low-res activity
    if len(expanded_segments) < 3:
        best_lr = max(unexpanded, key=lambda s: s["low_res"]["motion_score"] + s["low_res"]["brightness_change"])
        return f"expand {best_lr['id']}"

    # Default random exploration    
    import random
    return f"expand {random.choice(unexpanded)['id']}"

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: int) -> dict:
    task_name = f"task_{task_id}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME if USE_LLM else "heuristic")

    try:
        obs = api_reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"score": 0.0}

    steps = 0
    rewards_list = []
    final_score = 0.0
    final_reward = None
    done = False
    
    while not done and steps < 200:
        steps += 1
        
        # Decide action
        action_str = heuristic_decide(obs)  # For baseline simplicity we run heuristic
        
        # Execute action
        try:
            result = api_step(action_str)
            
            if "detail" in result: # HTTP Error handling
                log_step(step=steps, action=action_str, reward=0.0, done=False, error=str(result["detail"]))
                break
                
            obs = result["observation"]
            reward_obj = result.get("reward")
            info = result.get("info", {})
            done = result["done"]
            
            # Step reward formatting
            step_reward_float = 0.0
            if reward_obj and "score" in reward_obj:
                step_reward_float = float(reward_obj["score"])
                final_score = step_reward_float
            elif "step_reward_hint" in info:
                step_reward_float = float(info["step_reward_hint"])
            
            rewards_list.append(step_reward_float)
            
            log_step(step=steps, action=action_str, reward=step_reward_float, done=done, error=None)
            
            if done:
                break
        except Exception as e:
            log_step(step=steps, action=action_str, reward=0.0, done=True, error=str(e))
            break
            
        time.sleep(0.05)

    success = (final_score >= 0.80)
    log_end(success=success, steps=steps, score=final_score, rewards=rewards_list)

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": steps
    }


def main():
    for task_id in [1, 2, 3]:
        run_task(task_id)


if __name__ == "__main__":
    main()
