"""
preprocess.py - FALCON Scenario Generator
==========================================
Generates scenarios.json: a pool of DYNAMIC scenario variants per task.

Each call to POST /reset draws a RANDOM variant from the pool, making
every episode unique. The agent cannot memorize answers - it must learn
a genuine search strategy.

Pool layout:
  Task 1 (Robbery/Easy):       20 variants, 10% no-crime
  Task 2 (Shoplifting/Medium): 20 variants, 10% no-crime
  Task 3 (Fighting/Hard):      20 variants, 10% no-crime

Each variant differs in:
  - Ground truth crime frame (randomized within the video)
  - Red herring segment positions (randomized)
  - Feature vector noise seed (different statistical fingerprint)
  - Episode length (slight variation)

Run once before starting the server:
    python preprocess.py

Output: scenarios.json (at project root)
"""

import json
import math
import os
import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

FEATURE_DIM = 128
FPS = 30
VARIANTS_PER_TASK = 20
NO_CRIME_FRACTION = 0.10  # 10% of variants are clean videos


# ---------------------------------------------------------------------------
# Low-level feature helpers
# ---------------------------------------------------------------------------

def gaussian_bump(i, center, width, amplitude, noise=0.05):
    base = amplitude * math.exp(-0.5 * ((i - center) / width) ** 2)
    return float(np.clip(base + np.random.normal(0, noise), 0.0, 1.0))


def flat_noise(low=0.05, high=0.25):
    return float(np.random.uniform(low, high))


def make_lr_feature_vector(motion, brightness, person_count):
    vec = np.random.randn(FEATURE_DIM) * 0.1
    vec[0] = motion
    vec[1] = brightness
    vec[2] = person_count / 10.0
    return vec.tolist()


def make_hr_feature_vector(motion, brightness, person_count, is_crime):
    vec = np.random.randn(FEATURE_DIM) * 0.15
    vec[0] = motion * 1.3
    vec[1] = brightness * 1.3
    vec[2] = person_count / 10.0
    if is_crime:
        vec[3:8] = np.random.uniform(0.7, 1.0, size=5)
    else:
        vec[3:8] = np.random.uniform(0.0, 0.2, size=5)
    return vec.tolist()


def compute_partial_reward_hint(seg_idx, gt_segment, n_segments):
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


def compute_no_crime_hint(seg_idx, n_segments):
    """All hints are low for clean (no-crime) videos."""
    return round(float(np.random.uniform(0.0, 0.08)), 3)


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------

def build_segments(n_segments, gt_segment, crime_type, red_herrings, is_clean=False):
    """Build the full segment list for one scenario variant."""
    segments = []
    for i in range(n_segments):
        is_crime = (not is_clean) and (i == gt_segment)
        near_crime = (not is_clean) and abs(i - gt_segment) <= 3
        is_red_herring = i in red_herrings

        if crime_type == "robbery":
            if is_crime:
                motion = gaussian_bump(i, gt_segment, 2.0, 0.95, 0.04)
                brightness = gaussian_bump(i, gt_segment, 1.5, 0.90, 0.03)
                person_count = int(np.random.choice([6, 7, 8]))
            elif near_crime:
                motion = gaussian_bump(i, gt_segment, 3.0, 0.60, 0.06)
                brightness = gaussian_bump(i, gt_segment, 2.5, 0.55, 0.05)
                person_count = int(np.random.choice([4, 5, 6]))
            elif is_red_herring:
                motion = float(np.random.uniform(0.55, 0.78))
                brightness = float(np.random.uniform(0.40, 0.65))
                person_count = int(np.random.choice([3, 4]))
            else:
                motion = flat_noise(0.05, 0.30)
                brightness = flat_noise(0.05, 0.25)
                person_count = int(np.random.randint(1, 4))
            trajectory = "rapid_aggressive_movement" if is_crime else ("suspicious_loitering" if is_red_herring else "normal")

        elif crime_type == "shoplifting":
            if is_crime:
                motion = flat_noise(0.05, 0.15)
                brightness = flat_noise(0.08, 0.18)
                person_count = int(np.random.choice([1, 2]))
                trajectory = "slow_deliberate_item_concealment"
            elif near_crime:
                motion = flat_noise(0.06, 0.22)
                brightness = flat_noise(0.08, 0.20)
                person_count = int(np.random.choice([1, 2]))
                trajectory = "slow_movement_near_shelves"
            elif is_red_herring:
                motion = float(np.random.uniform(0.70, 0.92))
                brightness = float(np.random.uniform(0.55, 0.80))
                person_count = int(np.random.choice([7, 8, 9, 10]))
                trajectory = "normal_busy_area"
            else:
                motion = flat_noise(0.10, 0.35)
                brightness = flat_noise(0.10, 0.30)
                person_count = int(np.random.randint(1, 6))
                trajectory = "normal"

        elif crime_type == "fighting":
            # Three-phase: argument / shoving / fight
            argument_end = int(n_segments * 0.40)
            shove_end = gt_segment

            if i < argument_end:
                motion = flat_noise(0.08, 0.22)
                brightness = flat_noise(0.08, 0.18)
                person_count = 2
                trajectory = "stationary_argument"
            elif i < shove_end:
                progress = (i - argument_end) / max(1, shove_end - argument_end)
                motion = float(np.clip(0.18 + progress * 0.32 + np.random.normal(0, 0.05), 0.1, 0.72))
                brightness = flat_noise(0.12, 0.28)
                person_count = 2
                trajectory = "contact_shoving_escalation"
            elif is_crime:
                motion = float(np.random.uniform(0.75, 0.93))
                brightness = float(np.random.uniform(0.55, 0.78))
                person_count = 2
                trajectory = "first_punch_thrown"
            elif i < gt_segment + 60:
                motion = float(np.random.uniform(0.78, 0.97))
                brightness = float(np.random.uniform(0.48, 0.78))
                person_count = int(np.random.choice([2, 3, 4]))
                trajectory = "ongoing_fight"
            elif is_red_herring:
                motion = float(np.random.uniform(0.65, 0.85))
                brightness = float(np.random.uniform(0.40, 0.65))
                person_count = int(np.random.choice([3, 4]))
                trajectory = "loud_crowd_disturbance"
            else:
                motion = flat_noise(0.05, 0.28)
                brightness = flat_noise(0.08, 0.22)
                person_count = int(np.random.randint(1, 8))
                trajectory = "normal_crowd"
        else:
            motion = flat_noise()
            brightness = flat_noise()
            person_count = 2
            trajectory = "normal"

        # For clean videos, override hints with all-low values
        if is_clean:
            motion = flat_noise(0.05, 0.30)
            brightness = flat_noise(0.05, 0.25)
            person_count = int(np.random.randint(0, 5))
            trajectory = "normal"

        hint = (
            compute_no_crime_hint(i, n_segments)
            if is_clean
            else compute_partial_reward_hint(i, gt_segment, n_segments)
        )

        lr_vec = make_lr_feature_vector(motion, brightness, person_count)
        hr_vec = make_hr_feature_vector(motion, brightness, person_count, is_crime)

        segments.append({
            "id": f"segment_{i}",
            "start_frame": i * FPS,
            "end_frame": (i + 1) * FPS - 1,
            "low_res": {
                "motion_score": round(motion, 4),
                "brightness_change": round(brightness, 4),
                "person_count": person_count,
                "lr_feature_vector": lr_vec,
            },
            "expanded": {
                "person_trajectories": trajectory,
                "hr_feature_vector": hr_vec,
                "partial_reward_hint": hint,
            },
        })

    return segments


# ---------------------------------------------------------------------------
# Task variant generators
# ---------------------------------------------------------------------------

def generate_task_1_variants(n_variants):
    """
    Task 1 - Robbery (Easy)
    200 segments, budget=40.
    GT frame: random in [1800, 4200] (segments 60-140).
    Signal: strong - high motion + brightness spike.
    Each variant has 1-2 random red-herring segments.
    """
    variants = []
    n_no_crime = max(1, int(n_variants * NO_CRIME_FRACTION))

    for v in range(n_variants):
        is_clean = (v < n_no_crime)
        n_segments = 200
        budget = int(n_segments * 0.20)

        if is_clean:
            gt_segment = -1
            gt_frame = -1
        else:
            gt_segment = int(np.random.randint(40, 160))
            gt_frame = gt_segment * FPS

        # 1-2 random red herrings not near crime
        red_herrings = []
        for _ in range(np.random.randint(1, 3)):
            rh = int(np.random.randint(0, n_segments))
            if abs(rh - gt_segment) > 10:
                red_herrings.append(rh)

        segs = build_segments(n_segments, gt_segment, "robbery", red_herrings, is_clean)

        variants.append({
            "variant_id": v,
            "is_clean": is_clean,
            "ground_truth_frame": gt_frame,
            "ground_truth_segment": gt_segment,
            "total_frames": n_segments * FPS,
            "total_segments": n_segments,
            "budget": budget,
            "red_herring_segments": red_herrings,
            "segments": segs,
        })

    return variants


def generate_task_2_variants(n_variants):
    """
    Task 2 - Shoplifting (Medium)
    500 segments, budget=100.
    GT frame: random in [3000, 12000] (segments 100-400).
    Signal: Hidden in trajectory data. Motion is LOW.
    1-3 random red-herring segments with high motion (busy areas).
    """
    variants = []
    n_no_crime = max(1, int(n_variants * NO_CRIME_FRACTION))

    for v in range(n_variants):
        is_clean = (v < n_no_crime)
        n_segments = 500
        budget = int(n_segments * 0.20)

        if is_clean:
            gt_segment = -1
            gt_frame = -1
        else:
            gt_segment = int(np.random.randint(80, 420))
            gt_frame = gt_segment * FPS

        # 1-3 red herrings with high motion
        red_herrings = []
        for _ in range(np.random.randint(1, 4)):
            rh = int(np.random.randint(0, n_segments))
            if abs(rh - gt_segment) > 15:
                red_herrings.append(rh)

        segs = build_segments(n_segments, gt_segment, "shoplifting", red_herrings, is_clean)

        variants.append({
            "variant_id": v,
            "is_clean": is_clean,
            "ground_truth_frame": gt_frame,
            "ground_truth_segment": gt_segment,
            "total_frames": n_segments * FPS,
            "total_segments": n_segments,
            "budget": budget,
            "red_herring_segments": red_herrings,
            "segments": segs,
        })

    return variants


def generate_task_3_variants(n_variants):
    """
    Task 3 - Fighting (Hard)
    900 segments, budget=135 (15%).
    GT frame: random in [9000, 21000] (segments 300-700).
    Three-phase structure in every variant. Motion peaks AFTER crime starts.
    1-2 red herrings (loud crowd disturbances that look like fights).
    """
    variants = []
    n_no_crime = max(1, int(n_variants * NO_CRIME_FRACTION))

    for v in range(n_variants):
        is_clean = (v < n_no_crime)
        n_segments = 900
        budget = int(n_segments * 0.15)

        if is_clean:
            gt_segment = -1
            gt_frame = -1
        else:
            gt_segment = int(np.random.randint(300, 700))
            gt_frame = gt_segment * FPS

        # 1-2 red herrings (crowd disturbances far from crime)
        red_herrings = []
        for _ in range(np.random.randint(1, 3)):
            rh = int(np.random.randint(0, n_segments))
            if abs(rh - gt_segment) > 30:
                red_herrings.append(rh)

        segs = build_segments(n_segments, gt_segment, "fighting", red_herrings, is_clean)

        variants.append({
            "variant_id": v,
            "is_clean": is_clean,
            "ground_truth_frame": gt_frame,
            "ground_truth_segment": gt_segment,
            "total_frames": n_segments * FPS,
            "total_segments": n_segments,
            "budget": budget,
            "red_herring_segments": red_herrings,
            "segments": segs,
        })

    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")

    print("FALCON - Generating dynamic scenario pool ...")
    print(f"  {VARIANTS_PER_TASK} variants per task, {int(NO_CRIME_FRACTION*100)}% no-crime episodes")
    print()

    task1_variants = generate_task_1_variants(VARIANTS_PER_TASK)
    no_crime_1 = sum(1 for v in task1_variants if v["is_clean"])
    print(f"  Task 1 (Robbery):     {len(task1_variants)} variants "
          f"({no_crime_1} no-crime, segments=200, budget=40)")

    task2_variants = generate_task_2_variants(VARIANTS_PER_TASK)
    no_crime_2 = sum(1 for v in task2_variants if v["is_clean"])
    print(f"  Task 2 (Shoplifting): {len(task2_variants)} variants "
          f"({no_crime_2} no-crime, segments=500, budget=100)")

    task3_variants = generate_task_3_variants(VARIANTS_PER_TASK)
    no_crime_3 = sum(1 for v in task3_variants if v["is_clean"])
    print(f"  Task 3 (Fighting):    {len(task3_variants)} variants "
          f"({no_crime_3} no-crime, segments=900, budget=135)")

    scenarios = {
        "version": "2.0",
        "project": "FALCON",
        "description": (
            "Dynamic CCTV scenario pool for OpenEnv. "
            "Each /reset call draws a random variant so the agent cannot "
            "memorize answers - it must learn a generalizable search strategy. "
            "10% of episodes are clean (no crime): agent must use submit_no_crime."
        ),
        "tasks": [
            {
                "task_id": 1,
                "crime_type": "robbery",
                "difficulty": "easy",
                "description": (
                    "Parking-lot robbery. Strong signal: crime segment has high motion "
                    "and brightness spike. GT frame randomized each episode. "
                    "1-2 random red herrings per episode. 10% clean episodes."
                ),
                "variants": task1_variants,
            },
            {
                "task_id": 2,
                "crime_type": "shoplifting",
                "difficulty": "medium",
                "description": (
                    "Supermarket shoplifting. Hidden signal: crime has LOW motion "
                    "(deliberate slow movement). Only trajectory data reveals it. "
                    "1-3 high-motion red herrings per episode. GT frame randomized. "
                    "10% clean episodes."
                ),
                "variants": task2_variants,
            },
            {
                "task_id": 3,
                "crime_type": "fighting",
                "difficulty": "hard",
                "description": (
                    "Bar altercation with causal chain: argument -> shoving -> fight. "
                    "Motion peaks AFTER crime starts. Low-res scan looks uniform. "
                    "GT frame randomized. 1-2 crowd-disturbance red herrings. "
                    "10% clean episodes."
                ),
                "variants": task3_variants,
            },
        ],
    }

    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nscenarios.json written -> {output_path} ({size_mb:.1f} MB)")
    print(f"Total variants: {VARIANTS_PER_TASK * 3} across 3 tasks")
    print("Run `uvicorn server.main:app --reload` to start the environment server.")


if __name__ == "__main__":
    main()
