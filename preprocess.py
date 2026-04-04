"""
preprocess.py — FALCON Data Generator
======================================
Generates scenarios.json: synthetic CCTV segment features for all 3 tasks.
No real video files needed. Features mimic what a real frame-level extractor
(e.g. ViT-based model) would produce from 1fps thumbnail scans.

Run this first before starting the server:
    python preprocess.py

Output: scenarios.json (at the project root)
"""

import json
import math
import os
import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURE_DIM = 128  # low-res feature vector dimension (lr_feature_vector)
HR_FEATURE_DIM = 128  # high-res (expanded) feature vector dimension

# ---------------------------------------------------------------------------
# Helper: Gaussian bump centred at crime segment, scaled by amplitude
# ---------------------------------------------------------------------------

def gaussian_bump(i: int, center: float, width: float, amplitude: float, noise: float = 0.05) -> float:
    base = amplitude * math.exp(-0.5 * ((i - center) / width) ** 2)
    return float(np.clip(base + np.random.normal(0, noise), 0.0, 1.0))


def flat_noise(low: float = 0.05, high: float = 0.25) -> float:
    return float(np.random.uniform(low, high))


# ---------------------------------------------------------------------------
# Segment-level feature generators
# ---------------------------------------------------------------------------

def make_lr_feature_vector(motion: float, brightness: float, person_count: int) -> list:
    """
    128-dim synthetic low-res feature vector.
    First 3 dims encode the hand-crafted features; rest is noise.
    This mimics what a real ViT thumbnail encoder would output.
    """
    vec = np.random.randn(FEATURE_DIM) * 0.1
    vec[0] = motion
    vec[1] = brightness
    vec[2] = person_count / 10.0  # normalised
    return vec.tolist()


def make_hr_feature_vector(motion: float, brightness: float, person_count: int, is_crime: bool) -> list:
    """
    128-dim high-res (expanded) feature vector.
    Crime segments have a separable high-res signature.
    """
    vec = np.random.randn(HR_FEATURE_DIM) * 0.15
    vec[0] = motion * 1.3
    vec[1] = brightness * 1.3
    vec[2] = person_count / 10.0
    if is_crime:
        # Strong anomaly signature visible only at full fps
        vec[3:8] = np.random.uniform(0.7, 1.0, size=5)
    else:
        vec[3:8] = np.random.uniform(0.0, 0.2, size=5)
    return vec.tolist()


def compute_partial_reward_hint(seg_idx: int, gt_segment: int, n_segments: int) -> float:
    """
    Mirrors SASHA's intermediate reward signal.
    Returns 0.8–1.0 if the agent expands the crime segment or its neighbours,
    decaying smoothly to 0.0 far away.
    """
    dist = abs(seg_idx - gt_segment)
    if dist == 0:
        return round(float(np.random.uniform(0.9, 1.0)), 3)
    elif dist <= 2:
        return round(float(np.random.uniform(0.7, 0.9)), 3)
    elif dist <= 5:
        return round(float(np.random.uniform(0.4, 0.7)), 3)
    elif dist <= 15:
        return round(float(np.random.uniform(0.1, 0.4)), 3)
    else:
        return round(float(np.random.uniform(0.0, 0.1)), 3)


# ---------------------------------------------------------------------------
# Task-level generators
# ---------------------------------------------------------------------------

def generate_task_1_robbery():
    """
    Task 1 — Robbery (Easy)
    200 segments, budget=40 (20%)
    Ground truth frame: 1350 → segment 45 (1350 // 30 = 45)
    Signal: STRONG — crime segment has high motion_score AND high brightness_change.
    Optimal strategy: expand 1-2 highest motion segments → find crime.
    """
    n_segments = 200
    fps = 30
    gt_frame = 1350
    gt_segment = gt_frame // fps  # = 45
    budget = int(n_segments * 0.2)  # = 40

    segments = []
    for i in range(n_segments):
        is_crime = (i == gt_segment)
        near_crime = abs(i - gt_segment) <= 3

        # Strong, visible signal
        motion = gaussian_bump(i, gt_segment, width=2.0, amplitude=0.95, noise=0.04) if near_crime else flat_noise(0.05, 0.25)
        brightness = gaussian_bump(i, gt_segment, width=1.5, amplitude=0.90, noise=0.03) if near_crime else flat_noise(0.05, 0.20)
        person_count = int(np.random.choice([6, 7, 8]) if is_crime else np.random.choice([1, 2, 3]))

        lr_vec = make_lr_feature_vector(motion, brightness, person_count)
        hr_vec = make_hr_feature_vector(motion, brightness, person_count, is_crime)

        seg = {
            "id": f"segment_{i}",
            "start_frame": i * fps,
            "end_frame": (i + 1) * fps - 1,
            "low_res": {
                "motion_score": round(motion, 4),
                "brightness_change": round(brightness, 4),
                "person_count": person_count,
                "lr_feature_vector": lr_vec,
            },
            "expanded": {
                "person_trajectories": "suspicious_rapid_movement" if is_crime else "normal",
                "hr_feature_vector": hr_vec,
                "partial_reward_hint": compute_partial_reward_hint(i, gt_segment, n_segments),
            },
        }
        segments.append(seg)

    return {
        "task_id": 1,
        "crime_type": "robbery",
        "difficulty": "easy",
        "description": "Robbery in a parking lot. High motion and brightness spike marks the crime segment.",
        "total_frames": n_segments * fps,
        "total_segments": n_segments,
        "budget": budget,
        "ground_truth_frame": gt_frame,
        "ground_truth_segment": gt_segment,
        "segments": segments,
    }


def generate_task_2_shoplifting():
    """
    Task 2 — Shoplifting (Medium)
    500 segments, budget=100 (20%)
    Ground truth frame: 6840 → segment 228
    Signal: HIDDEN — crime in person_trajectories (slow, deliberate movement).
    Red herring: segment 150 (busy checkout) has high motion but is normal.
    """
    n_segments = 500
    fps = 30
    gt_frame = 6840
    gt_segment = gt_frame // fps  # = 228
    budget = int(n_segments * 0.2)  # = 100

    # Red herring segment (busy checkout lane)
    red_herring_segment = 150

    segments = []
    for i in range(n_segments):
        is_crime = (i == gt_segment)
        near_crime = abs(i - gt_segment) <= 4
        is_red_herring = (i == red_herring_segment)

        # Shoplifter moves slowly — LOW motion_score (the trap for naive agents)
        if is_crime:
            motion = flat_noise(0.05, 0.15)          # suspiciously low
            brightness = flat_noise(0.08, 0.18)
            person_count = 2
            trajectory = "slow_deliberate_item_concealment"
        elif is_red_herring:
            motion = float(np.random.uniform(0.75, 0.90))   # HIGH but normal
            brightness = float(np.random.uniform(0.60, 0.80))
            person_count = int(np.random.choice([8, 9, 10]))
            trajectory = "normal_busy_checkout"
        elif near_crime:
            motion = flat_noise(0.08, 0.20)
            brightness = flat_noise(0.08, 0.18)
            person_count = int(np.random.choice([1, 2]))
            trajectory = "slow_movement_near_shelves"
        else:
            motion = flat_noise(0.10, 0.35)
            brightness = flat_noise(0.10, 0.30)
            person_count = int(np.random.randint(1, 6))
            trajectory = "normal"

        lr_vec = make_lr_feature_vector(motion, brightness, person_count)
        hr_vec = make_hr_feature_vector(motion, brightness, person_count, is_crime)

        seg = {
            "id": f"segment_{i}",
            "start_frame": i * fps,
            "end_frame": (i + 1) * fps - 1,
            "low_res": {
                "motion_score": round(motion, 4),
                "brightness_change": round(brightness, 4),
                "person_count": person_count,
                "lr_feature_vector": lr_vec,
            },
            "expanded": {
                "person_trajectories": trajectory,
                "hr_feature_vector": hr_vec,
                "partial_reward_hint": compute_partial_reward_hint(i, gt_segment, n_segments),
            },
        }
        segments.append(seg)

    return {
        "task_id": 2,
        "crime_type": "shoplifting",
        "difficulty": "medium",
        "description": "Shoplifter slowly conceals items near aisle 7. Motion is low; trajectory data reveals the crime. Busy checkout (segment 150) is a red herring.",
        "total_frames": n_segments * fps,
        "total_segments": n_segments,
        "budget": budget,
        "ground_truth_frame": gt_frame,
        "ground_truth_segment": gt_segment,
        "segments": segments,
    }


def generate_task_3_fighting():
    """
    Task 3 — Fighting (Hard)
    900 segments, budget=135 (15% — tightest)
    Ground truth frame: 10620 → segment 354 (first punch)
    Three-phase structure: argument (0-300), shoving (301-354), fight (354+)
    Signal: REQUIRES CAUSAL CHAIN — motion peaks AFTER crime starts.
             Low-res scan looks uniform. Agent must expand argument phase to understand context.
    """
    n_segments = 900
    fps = 30
    gt_frame = 10620
    gt_segment = gt_frame // fps  # = 354
    budget = int(n_segments * 0.15)  # = 135 (tightest budget)

    # Phase boundaries
    argument_end = 300      # segments 0-300: two people arguing, mild motion
    shove_end = 354         # segments 301-354: shoving, moderate motion
    # segments 354+: full fight, HIGH motion

    segments = []
    for i in range(n_segments):
        is_crime = (i == gt_segment)

        if i < argument_end:
            # Phase 1: Argument — looks calm from low-res
            motion = flat_noise(0.10, 0.25)
            brightness = flat_noise(0.10, 0.20)
            person_count = 2
            trajectory = "stationary_argument"

        elif i < shove_end:
            # Phase 2: Shoving — borderline, some motion
            progress = (i - argument_end) / (shove_end - argument_end)
            motion = float(np.clip(0.20 + progress * 0.30 + np.random.normal(0, 0.05), 0.1, 0.7))
            brightness = flat_noise(0.15, 0.30)
            person_count = 2
            trajectory = "contact_shoving_escalation"

        elif i == gt_segment:
            # Crime segment: first punch — motion spikes HERE
            motion = float(np.random.uniform(0.75, 0.92))
            brightness = float(np.random.uniform(0.55, 0.75))
            person_count = 2
            trajectory = "first_punch_thrown"

        elif i < gt_segment + 50:
            # Post-crime: fight continues, HIGH motion (misleads naive max-motion agent)
            motion = float(np.random.uniform(0.80, 0.98))
            brightness = float(np.random.uniform(0.50, 0.80))
            person_count = int(np.random.choice([2, 3, 4]))  # bystanders join
            trajectory = "ongoing_fight"

        else:
            # Other segments: normal crowd
            motion = flat_noise(0.05, 0.30)
            brightness = flat_noise(0.10, 0.25)
            person_count = int(np.random.randint(1, 8))
            trajectory = "normal_crowd"

        lr_vec = make_lr_feature_vector(motion, brightness, person_count)
        hr_vec = make_hr_feature_vector(motion, brightness, person_count, is_crime)

        seg = {
            "id": f"segment_{i}",
            "start_frame": i * fps,
            "end_frame": (i + 1) * fps - 1,
            "low_res": {
                "motion_score": round(motion, 4),
                "brightness_change": round(brightness, 4),
                "person_count": person_count,
                "lr_feature_vector": lr_vec,
            },
            "expanded": {
                "person_trajectories": trajectory,
                "hr_feature_vector": hr_vec,
                "partial_reward_hint": compute_partial_reward_hint(i, gt_segment, n_segments),
            },
        }
        segments.append(seg)

    return {
        "task_id": 3,
        "crime_type": "fighting",
        "difficulty": "hard",
        "description": (
            "Bar altercation escalating over 15 minutes. Three phases: "
            "argument (segs 0-300), shoving (segs 301-353), fight (seg 354+). "
            "Crime starts at frame 10620 (first punch). Low-res scan looks uniform throughout. "
            "Motion peaks AFTER the crime starts — naive max-motion agent will be off by ~30s."
        ),
        "total_frames": n_segments * fps,
        "total_segments": n_segments,
        "budget": budget,
        "ground_truth_frame": gt_frame,
        "ground_truth_segment": gt_segment,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = os.path.join(os.path.dirname(__file__), "scenarios.json")

    print("FALCON — Generating scenarios.json ...")

    task1 = generate_task_1_robbery()
    print(f"  ✅ Task 1 (Robbery):     {task1['total_segments']} segments, budget={task1['budget']}, GT frame={task1['ground_truth_frame']}")

    task2 = generate_task_2_shoplifting()
    print(f"  ✅ Task 2 (Shoplifting): {task2['total_segments']} segments, budget={task2['budget']}, GT frame={task2['ground_truth_frame']}")

    task3 = generate_task_3_fighting()
    print(f"  ✅ Task 3 (Fighting):    {task3['total_segments']} segments, budget={task3['budget']}, GT frame={task3['ground_truth_frame']}")

    scenarios = {
        "version": "1.0",
        "project": "FALCON",
        "description": "Synthetic CCTV segment features for OpenEnv hackathon. Based on UCF-Crime dataset structure.",
        "tasks": [task1, task2, task3],
    }

    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n📦 scenarios.json written → {output_path} ({size_mb:.1f} MB)")
    print("   Run `uvicorn server.main:app --reload` to start the environment server.")


if __name__ == "__main__":
    main()
