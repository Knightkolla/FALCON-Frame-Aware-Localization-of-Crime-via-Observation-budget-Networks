---
title: FALCON - CCTV Crime Localization Environment
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - surveillance
  - crime-detection
  - cctv
  - temporal-sampling
  - pytorch
license: mit
short_description: "CCTV crime frame localization under 20% budget"
---

# FALCON - Frame-Aware Localization of Crime via Observation-budget Networks

> **Meta PyTorch OpenEnv Hackathon** | Scaler School of Technology × Hugging Face  
> Built by **Dhavala Kartikeya Somayaji** | Deadline: April 8, 2024

---

## One-Sentence Pitch

*FALCON proves that 20% of temporal segments are enough to find the exact frame where a crime begins in hours of CCTV footage, inspired by how medical AI diagnoses cancer from tiny fractions of gigapixel slides.*

---

## What Is FALCON?

FALCON is an OpenEnv-compliant reinforcement learning environment where an AI agent must locate the **exact frame** a crime starts in a CCTV surveillance video — using only **20% of the video's temporal segments**.

The problem mirrors digital pathology: just as tumours occupy a tiny fraction of a gigapixel slide, crime events occupy a tiny fraction of hours of surveillance footage. FALCON uses budget-constrained sequential sampling with deep RL to solve this in the temporal domain.

---

## Architecture Mapping

| Medical AI (Inspiration) | FALCON (Crime Detection) |
|---|---|
| Whole slide image (gigapixels) | Full CCTV video (hours) |
| 1fps thumbnail scan of slide | 1fps thumbnail of video |
| N spatial patches across slide | N temporal segments across timeline |
| "Zoom into" selected patch | "Expand" selected segment to 30fps |
| Budget: 20% of patches | Budget: 20% of segments |
| HAFED hierarchical attention | Temporal attention (same architecture) |
| TSU: cosine-similar patch update | TSU: cosine + temporally adjacent update |
| Output: cancer YES/NO | Output: exact crime start frame |

---

## Architecture

```
Agent (LLM or PPO)
       │
       │ text action: "expand segment_42"
       ▼
┌─────────────────────────────────┐
│       FastAPI Server            │
│  POST /reset  POST /step        │
│  GET  /state  GET  /health      │
└────────────┬────────────────────┘
             │
     ┌───────▼────────┐
     │  CCTVCosineEnv │  ← FALCON's RL environment
     │                │
     │  step():       │
     │  1. Mark segment visited        │
     │  2. Cosine similarity (TSU)     │
     │  3. Adjacent segment update     │   ← FALCON addition
     │  4. FGlobal MLP state update    │   ← FALCON TSU module
     │  5. Return partial_reward_hint  │   ← replaces -CrossEntropy
     └───────┬─────────┘
             │
     ┌───────▼─────────┐
     │   scenarios.json │  ← synthetic CCTV features (preprocess.py)
     └──────────────────┘
```

---

## Project Structure

```
FALCON/
├── inference.py              ← LLM agent (must be named exactly this)
├── preprocess.py             ← Generates scenarios.json
├── scenarios.json            ← Pre-baked scenario data (3 tasks)
├── openenv.yaml              ← OpenEnv spec metadata
├── Dockerfile                ← HF Spaces container
├── requirements.txt
├── README.md
│
├── server/
│   ├── main.py               ← FastAPI: /reset /step /state /health
│   ├── models.py             ← Pydantic types (OpenEnv compliant)
│   ├── grader.py             ← Deterministic frame proximity reward
│   ├── data_loader.py        ← Loads scenarios.json
│   └── CCTV_cosine_env.py   ← FALCON RL environment
│
└── falcon_core/              ← Core RL/TSU modules
    ├── modules/fglobal_mlp.py ← FGlobal MLP (TSU)
    ├── rl_algorithms/ppo.py   ← PPO Actor/Critic/Agent — reused as-is
    └── ...
```

---

## The Three Tasks

### Task 1 — Robbery (Easy)
| Property | Value |
|---|---|
| Segments | 200 (budget: 40) |
| Ground Truth Frame | 1,350 |
| Signal | **Strong**: high `motion_score` + `brightness_change` in crime segment |
| Strategy | Expand 1-2 highest motion segments → find crime |
| Target Score | 1.0 (exact) or 0.9 (within 1s) |

### Task 2 — Shoplifting (Medium)
| Property | Value |
|---|---|
| Segments | 500 (budget: 100) |
| Ground Truth Frame | 6,840 |
| Signal | **Hidden**: only `person_trajectories` reveals it. Motion is LOW (shoplifter moves slow). Red herring: segment 150 (busy checkout) has high motion. |
| Strategy | Must explore trajectory data, ignore `motion_score` |
| Target Score | 0.9 within 1s, 0.7 within 3s |

### Task 3 — Fighting (Hard)
| Property | Value |
|---|---|
| Segments | 900 (budget: 135) |
| Ground Truth Frame | 10,620 (first punch) |
| Signal | **Causal chain**: argument → shoving → fight. Motion peaks *after* crime starts. Low-res scan looks uniform throughout. |
| Strategy | Agent must expand the argument phase to understand context, then locate exact transition |
| Target Score | 0.9 within 1s, 0.5 within 5s, 0.1 within 30s |

---

## Reward Function

| Frame Distance | Score |
|---|---|
| Exact match | **1.00** |
| Within 1 second (30 frames) | **0.90** + efficiency bonus |
| Within 3 seconds (90 frames) | **0.70** + efficiency bonus |
| Within 10 seconds (300 frames) | **0.40** |
| Within 30 seconds (900 frames) | **0.10** |
| Beyond 30 seconds | **0.00** |

**Efficiency bonus**: up to +0.10 for using fewer budget expansions (% of budget unused × 0.10 × base_score).

---

## Quick Start

### 1. Generate Data
```bash
python preprocess.py
```

### 2. Start Environment Server
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run Inference (heuristic baseline, no API key needed)
```bash
python inference.py
```

### 4. Run with LLM Agent
```bash
OPENAI_API_KEY=sk-... python inference.py
```

### 5. Docker
```bash
docker build -t falcon .
docker run -p 7860:7860 falcon
```

---

## API Reference

### `POST /reset?task_id={1,2,3}`
Start a new episode.

```json
// Response: Observation
{
  "task_id": 1,
  "crime_type": "robbery",
  "total_segments": 200,
  "budget_remaining": 40,
  "budget_total": 40,
  "episode_done": false,
  "segments": [
    {
      "id": "segment_0",
      "start_frame": 0,
      "end_frame": 29,
      "low_res": { "motion_score": 0.12, "brightness_change": 0.08, "person_count": 1 },
      "is_expanded": false,
      "expanded": null
    },
    ...
  ]
}
```

### `POST /step`
Take an action. Request body: `{"raw": "expand segment_45"}`

Valid actions:
- `expand segment_{N}` — expand segment N (costs 1 budget)
- `flag segment_{N}` — flag as suspicious (free)
- `submit_frame {F}` — submit answer frame F (ends episode)
- `submit_no_crime` — declare clean video (ends episode)

### `GET /state`
Current episode state summary.

### `GET /health`
Liveness check. Returns `{"status": "ok"}`.

---

## Dataset

FALCON uses **synthetically generated** features inspired by the **UCF-Crime** dataset:
- 128-hour surveillance video dataset
- 1,900 videos across 13 crime categories
- Frame-level annotations

No real video files are shipped. Feature vectors are generated by `preprocess.py` to match the statistical properties of real frame-level encodings.

**Citation:**
> Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. *CVPR 2018*.

---

## Architecture Inspiration

FALCON's architecture draws from research in budget-constrained sequential sampling for medical imaging, adapted for the temporal video domain.

Key components in `falcon_core/`:
- **FGlobal MLP** (`modules/fglobal_mlp.py`) — Targeted State Updater
- **PPO Actor/Critic/Agent** (`rl_algorithms/ppo.py`) — RL policy
- **Cosine similarity environment** — adapted for temporal segments

---

## OpenEnv Compliance Checklist

- [x] Pydantic typed models for Observation, Action, Reward
- [x] `step()` / `reset()` / `state()` endpoints implemented
- [x] `openenv.yaml` with full metadata
- [x] Minimum 3 tasks (easy/medium/hard)
- [x] Graders scoring 0.0–1.0
- [x] Partial reward signal (partial_reward_hint per step)
- [x] Baseline `inference.py` at root
- [x] Working `Dockerfile`
- [x] Full README documentation
- [x] Runtime < 20 minutes (< 5 min for 3 tasks)
- [x] CPU-only (vcpu=2, memory=8GB compatible)
