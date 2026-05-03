# BC->PPO Alignment Fix and Rerun

## 1. Fixed defects in BC->PPO transfer

Two transfer mismatches were present in the original pipeline:

1. BC consumed normalized observations, while PPO consumed raw observations.
2. BC used a `Linear -> Tanh` action head, while PPO used a linear Gaussian mean head.

Fixes applied:

- BC fine-tuning environments and evaluation now use the BC checkpoint's `obs_mean` / `obs_std`.
- BC KL regularization now assumes normalized observations consistently.
- BC training now defaults to a **linear output head** so the copied BC actor exactly matches the PPO action head.
- PPO initialization now rejects old tanh-headed BC checkpoints.

Relevant files:

- `src/ilrl_lab/ppo_training.py`
- `scripts/fine_tune_ppo_from_bc.py`
- `scripts/train_ppo.py`
- `src/ilrl_lab/bc.py`
- `scripts/train_bc.py`

## 2. Sanity check

Sanity check artifact:

- `artifacts/checks/bc_ppo_alignment_clean50_fixed.json`

Result:

- `mean_action_l2_diff = 0.0`
- `mean_cosine_similarity = 1.0`

This confirms that, after the fix, the BC policy and the PPO actor are functionally identical at step 0.

## 3. Clean 50-demo matched rerun

Aligned BC checkpoint:

- `artifacts/checkpoints/bc_aligned/clean50/bc_20260504_022538/checkpoint.pt`

3-seed final 30-episode mean:

- Scratch PPO:
  - success `0.00`
  - return `21.15`
  - final distance `0.732`

- BC+PPO:
  - success `0.111`
  - return `81.83`
  - final distance `0.476`

Conclusion:

- After fixing the transfer path, the core claim still holds.
- BC-initialized PPO remains clearly better than scratch PPO on the detour task.

## 4. Quality and quantity reruns

### BC-only rerun

- `10 demos`: success `0.02`, final distance `0.747`
- `clean 50 demos`: success `0.28`, final distance `0.329`
- `noisy 50 demos`: success `0.66`, final distance `0.292`
- `200 demos`: success `0.94`, final distance `0.075`

### BC+PPO final 3-seed mean

- `clean 50 demos`:
  - success `0.111`
  - final distance `0.476`

- `noisy 50 demos`:
  - success `0.00`
  - final distance `0.677`

- `10 demos`:
  - success `0.00`
  - final distance `0.704`

- `200 demos`:
  - success `0.10`
  - final distance `0.426`

## Updated takeaways

1. The original transfer path had a real implementation defect, and the concern was valid.
2. After fixing that defect, the main detour result survives:
   - clean-demo BC+PPO outperforms scratch PPO.
3. The noisy-demo story also survives, and becomes sharper:
   - noisy demos improve BC-only robustness,
   - but they are worse as PPO warm-start priors.
4. The quantity story changes slightly:
   - `10 demos` is still too weak,
   - `50 demos` remains a strong warm-start regime,
   - `200 demos` becomes competitive again after the fix, because the aligned BC prior is now much stronger.
