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

API_KEY = os.environ.get("API_KEY", "")
# LLM Router URL
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini") # Fallback to a common testing model if missing

# Environment Server URL
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "falcon_cctv"

try:
    from openai import OpenAI
    # The client connects strictly to the LiteLLM proxy
    client = OpenAI(
        api_key=os.environ["API_KEY"], 
        base_url=os.environ["API_BASE_URL"]
    )
    USE_LLM = True
except Exception:
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
# ---------------------------------------------------------------------------
# LLM Logic
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert CCTV analyst investigating an incident.
You must find the EXACT FRAME a crime starts using a limited observation budget.
You will be provided with N segments (each 1 second long). 
Your initial observation shows only low-resolution info (motion_score, person_count).

Choose one of these precise actions:
- "expand segment_X": reveals high-res trajectory and a partial_reward_hint >0 if close to crime. Costs budget.
- "submit_frame X": if you are highly confident, guess the exact frame timestamp!
- "submit_no_crime": if you are 100% sure the video contains no crime.

Reply with EXACTLY ONE ACTION on a single line. No quotes, no markdown, no reasoning.
"""

def get_llm_action(obs: dict, history: list) -> str:
    # Summarize state for LLM
    prompt = f"Crime type: {obs['crime_type']} ({obs['difficulty']}). Budget left: {obs['budget_remaining']} / {obs['budget_total']}."
    summary = []
    for s in obs["segments"]:
        if s["is_expanded"]:
            summary.append(f"[{s['id']}] Frame {s['start_frame']} - HR: {s['expanded']['partial_reward_hint']:.2f}")
        else:
            summary.append(f"[{s['id']}] Frame {s['start_frame']} - LR motion: {s['low_res']['motion_score']:.2f}")
    
    prompt += "\nSegments:\n" + "\n".join(summary)
    if history:
        prompt += f"\nPrevious steps:\n" + "\n".join(history[-3:])
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=60,
        )
        action = (response.choices[0].message.content or "").strip()
        return action if action else f"expand segment_0"
    except Exception as e:
        print(f"[DEBUG] Model Request Failed: {e}", flush=True)
        return f"expand segment_0"
    
# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: int) -> dict:
    task_name = f"task_{task_id}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = api_reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.01, rewards=[])
        return {"score": 0.01}

    steps = 0
    history = []
    rewards_list = []
    final_score = 0.01
    final_reward = None
    done = False
    
    while not done and steps < 200:
        steps += 1
        
        # Decide action strictly using LLM
        action_str = get_llm_action(obs, history)
        
        # Execute action
        try:
            result = api_step(action_str)
            
            if "detail" in result: # HTTP Error handling
                log_step(step=steps, action=action_str, reward=0.01, done=False, error=str(result["detail"]))
                break
                
            obs = result["observation"]
            reward_obj = result.get("reward")
            info = result.get("info", {})
            done = result["done"]
            
            # Step reward formatting
            step_reward_float = 0.01
            if reward_obj and "score" in reward_obj:
                step_reward_float = float(reward_obj["score"])
                final_score = step_reward_float
            elif "step_reward_hint" in info:
                step_reward_float = float(info["step_reward_hint"])
            
            rewards_list.append(step_reward_float)
            
            history.append(f"Step {steps}: {action_str} -> reward {step_reward_float:.2f}")
            log_step(step=steps, action=action_str, reward=step_reward_float, done=done, error=None)
            
            if done:
                break
        except Exception as e:
            log_step(step=steps, action=action_str, reward=0.01, done=True, error=str(e))
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
    if not API_KEY:
        print("[DEBUG] WARNING: Empty API_KEY. Script might fail calling HF router or proxy.", flush=True)
        
    for task_id in [1, 2, 3]:
        run_task(task_id)


if __name__ == "__main__":
    main()
