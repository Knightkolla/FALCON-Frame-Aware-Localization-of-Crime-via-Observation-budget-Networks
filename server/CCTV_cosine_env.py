"""
server/CCTV_cosine_env.py — FALCON Environment
===============================================
Direct adaptation of FALCON's WSI_cosine_env.py for temporal video segments.

What changed vs. FALCON original (3 logical diffs):
  1. Class renamed: WSICosineObservationEnv → CCTVCosineEnv
  2. Reward: -CrossEntropyLoss(classify(state)) → frame_proximity_hint(action)
  3. Step: also updates temporally adjacent segments (in addition to cosine-similar ones)

Everything else — budget tracking, TSU cosine update, visited_patches mask,
done condition — is copied verbatim from FALCON.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class CCTVCosineEnv:
    """
    ### Description
    FALCON's RL environment. Adapted from FALCON's WSICosineObservationEnv.

    A CCTV video is represented as N temporal segments. The agent observes
    low-res (1fps thumbnail) feature vectors for all segments and must
    strategically "expand" up to `budget` segments to full 30fps resolution
    in order to find the exact frame where a crime starts.

    ### Observation Space
    State tensor of shape (1, N, 128): low-res feature vectors for all segments.
    Expanded segments have their lr vector replaced by the hr vector (same as FALCON).

    ### Action Space
    Discrete integer in [0, N-1]: the index of the segment to expand next.

    ### Transition Dynamics (mirrors FALCON's TSU update)
    When segment a_t is expanded:
      1. Compute cosine similarity of a_t with all unvisited segments.
      2. Segments with similarity >= cosine_threshold get their state vector
         updated via the FGlobal MLP (TSU): state[j] = fglobal(v_at, z_at, state[j])
      3. Temporally adjacent segments (±1, ±2) also get TSU updates (FALCON addition).
      4. Budget counter decrements by 1.

    ### Reward
    Intermediate reward = partial_reward_hint[action] (0-1, from scenarios.json).
    This replaces FALCON's -CrossEntropyLoss, giving the agent continuous feedback.
    Final reward is computed by grader.py when agent submits.

    ### Episode End
    Episode ends when budget is exhausted (agent must then submit).
    """

    def __init__(
        self,
        lr_features: torch.Tensor,       # (1, N, 128) — low-res feature state
        hr_features: torch.Tensor,        # (N, 128)    — high-res ground truth
        partial_hints: list,              # N floats    — intermediate reward signal
        ground_truth_frame: int,          # int         — crime start frame (for info)
        conf,                             # config namespace (same fields as FALCON conf)
    ):
        self.conf = conf
        self.ground_truth_frame = ground_truth_frame
        self.partial_hints = partial_hints  # list[float] length N

        # --- State (mirrors FALCON exactly) ---
        self.features = lr_features.clone()         # initial lr state, never mutated
        self.current_state = lr_features.clone()    # mutable state updated each step
        self.hr_features = hr_features              # (N, 128) full-res counterparts

        self.B, self.N, self.d = self.current_state.shape  # 1, N, 128
        self.current_time_step = 0
        self.max_time_steps = math.ceil(self.N * conf.frac_visit)

        # Tracking (same as FALCON)
        self.visited_patches = torch.zeros(self.N, dtype=torch.bool)
        self.visited_patch_idx: list = []
        self.reward_history: list = []
        self.mask = torch.ones(self.N)
        self.cosine_threshold = conf.cosine_threshold
        self.high_cosine_indices: Optional[torch.Tensor] = None
        self.visited_all_patches = False

    # -------------------------------------------------------------------------
    def reset(self):
        """Reset episode state. Mirrors FALCON's reset() exactly."""
        self.current_state = self.features.clone()
        self.current_time_step = 0
        self.visited_patches = torch.zeros(self.N, dtype=torch.bool)
        self.visited_patch_idx = []
        self.reward_history = []
        self.mask = torch.ones(self.N)
        self.visited_all_patches = False

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        action: torch.Tensor,       # segment index to expand
        state_update_net,           # FGlobal MLP (TSU) — same as FALCON
        device: str,
    ):
        """
        Expand segment `action`, update state via TSU, return (state, reward, done).

        Mirrors FALCON's WSICosineObservationEnv.step() with two changes:
          - reward = partial_hints[action]  (instead of -CrossEntropy)
          - adjacent segment bonus update   (temporal locality, FALCON addition)
        """

        # --- All patches visited: keep stepping until max_time_steps ---
        if self.visited_all_patches:
            self.current_time_step += 1
            done = self.current_time_step == self.max_time_steps
            reward = self.partial_hints[action.item()] if not done else self.partial_hints[action.item()]
            self.reward_history.append(reward)
            if done:
                final_state = self.current_state.clone()
                self.reset()
                return final_state, reward, True
            return self.current_state.clone(), reward, False

        # --- Normal step (mirrors FALCON lines 88–143) ---
        if self.current_time_step < self.max_time_steps:
            self.current_time_step += 1
            act_idx = action.item()

            # Mark as visited
            self.visited_patches[act_idx] = True
            self.visited_patch_idx.append(act_idx)

            # Valid (unvisited) indices
            valid_indices = (self.mask == 1).nonzero()[:, 0]

            # --- Step 1: Cosine similarity (FALCON TSU, verbatim) ---
            cosine_vector = torch.cosine_similarity(
                self.current_state[0], self.current_state[0][act_idx]
            )
            high_cosine_indices = (
                torch.abs(cosine_vector) >= self.cosine_threshold
            ).nonzero()[:, 0]

            # --- Step 2: Mask visited cosine-similar patches ---
            self.mask[high_cosine_indices] = 0
            self.visited_patches[high_cosine_indices] = True

            # Remove the expanded segment itself
            high_cosine_indices = high_cosine_indices[high_cosine_indices != act_idx]
            high_cosine_indices = torch.tensor(
                list(set(high_cosine_indices.tolist()).intersection(valid_indices.tolist())),
                dtype=torch.long,
                device=device,
            )

            # --- FALCON addition: adjacent temporal segments ---
            adjacent = [
                i for i in [act_idx - 2, act_idx - 1, act_idx + 1, act_idx + 2]
                if 0 <= i < self.N
                and not self.visited_patches[i].item()
                and i in valid_indices.tolist()
            ]
            adj_tensor = torch.tensor(adjacent, dtype=torch.long, device=device)

            # Combine cosine + adjacent (deduplicated)
            if len(high_cosine_indices) > 0 or len(adj_tensor) > 0:
                parts = [t for t in [high_cosine_indices, adj_tensor] if len(t) > 0]
                all_update_indices = torch.unique(torch.cat(parts)).long()
            else:
                all_update_indices = torch.tensor([], dtype=torch.long, device=device)

            self.high_cosine_indices = all_update_indices

            # --- Step 3: TSU update (verbatim from FALCON step() lines 107–120) ---
            v_at = self.hr_features[act_idx].to(device)   # expanded hr feature

            if all_update_indices.shape[0] == 0:
                # No similar/adjacent patches — just update the expanded one
                self.current_state[0][act_idx] = v_at
            else:
                ui = all_update_indices.long()   # ensure long dtype for indexing
                # FGlobal MLP: [v_at ‖ z_at ‖ state[j]] → predicted_hr[j]
                input_f_global = torch.cat(
                    (
                        v_at.repeat(len(ui), 1),
                        self.current_state[0][act_idx].repeat(len(ui), 1),
                        self.current_state[0][ui],
                    ),
                    dim=1,
                )
                self.current_state = self.current_state.clone()
                self.current_state[:, ui, :] = state_update_net(input_f_global)

            # Replace all visited segments with their actual hr features
            vpi = torch.tensor(self.visited_patch_idx, dtype=torch.long)
            self.current_state[0][vpi] = self.hr_features[vpi]

            # --- Reward: partial_reward_hint (replaces FALCON's -CrossEntropy) ---
            reward = self.partial_hints[act_idx]
            self.reward_history.append(reward)

            done = False

            # Check budget exhausted
            if self.current_time_step == self.max_time_steps:
                done = True
                final_state = self.current_state.clone()
                self.reset()
                return final_state, reward, done

            # Check if all patches visited
            if self.visited_patches.sum().item() == self.N:
                self.visited_all_patches = True

        return self.current_state.clone(), reward, done

    # -------------------------------------------------------------------------
    def get_similar_patches(self) -> Optional[torch.Tensor]:
        """Return the indices of patches updated in the last TSU step."""
        return self.high_cosine_indices

    def budget_remaining(self) -> int:
        """How many expand steps are left."""
        return self.max_time_steps - self.current_time_step

    def is_done(self) -> bool:
        return self.current_time_step >= self.max_time_steps
