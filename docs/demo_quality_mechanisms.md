# Demo Quality Mechanisms

This note explains **why clean and noisy demonstrations behave differently** after the BC -> PPO transfer fix.

Artifacts:

- summary JSON: `artifacts/figures/demo_quality_mechanisms/summary.json`
- figure: `artifacts/figures/demo_quality_mechanisms/mechanism_overview.png`

## 1. Demo action variance

The noisy dataset is genuinely much rougher at the action level.

- clean `mean_delta_action_norm`: `0.0247`
- noisy `mean_delta_action_norm`: `0.3889`

So the noise injection is not cosmetic. It creates much less smooth action sequences.

## 2. BC action spread on shared eval states

Surprisingly, noisy BC is **not** simply a much broader policy everywhere.

- clean BC `mean_pairwise_l2_to_mean`: `0.663`
- noisy BC `mean_pairwise_l2_to_mean`: `0.642`

That means the BC-only robustness gain from noisy data is not just “higher entropy” in the naive sense.

## 3. Clean vs noisy BC on the same states

The two BC policies are still highly aligned directionally.

- mean action L2 diff: `0.187`
- mean cosine similarity: `0.990`

So noisy BC is not learning a totally different strategy. It is making more local magnitude and correction changes within a very similar detour template.

## 4. Early PPO divergence from the BC prior

After the transfer fix, the PPO-vs-BC comparison is finally meaningful at step 0.

Step-0 alignment is near exact:

- clean prior: mean L2 from BC `0.019`
- noisy prior: mean L2 from BC `0.002`

Both are effectively identical up to tiny numerical differences.

After that, PPO drifts away from the BC prior in both cases.

### Clean prior

- 10k: mean L2 `0.400`
- 20k: mean L2 `0.533`
- 50k: mean L2 `0.869`

### Noisy prior

- 10k: mean L2 `0.365`
- 20k: mean L2 `0.506`
- 50k: mean L2 `0.878`

The short-run divergence magnitudes are similar by 50k, but their early training behavior differs:

- clean short run already shows a small success signal at `20k`
- noisy short run collapses after a good `10k` eval and becomes less stable

## Interpretation

The corrected mechanism story is:

1. Noisy demos create much rougher action labels.
2. That roughness does **not** simply turn BC into a high-entropy policy.
3. Instead, noisy BC appears to preserve the same overall detour direction while changing local correction behavior.
4. Those local corrections help **BC-only robustness**.
5. But for PPO warm starts, the **clean prior is sharper and more useful**, so downstream RL performs better.

## Bottom line

> Noisy demonstrations help BC by broadening local recovery behavior, but clean demonstrations provide a better structured prior for downstream PPO fine-tuning.
