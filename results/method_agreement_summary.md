# Cross-Method Explainability Validation

Five explainability methods (SHAP, Permutation Importance, ALE, LIME,
Counterfactual Search) were applied to models trained on a Cu2O
photocatalysis dataset with a known ground truth: Activity is driven by
light intensity, Selectivity is driven by LED color, and Temperature_Rise_C
is a deliberate decoy with zero true relationship to either target.

| Feature | Target | SHAP | Permutation | ALE | LIME | Conclusion |
|---|---|---|---|---|---|---|
| Light_Intensity | Activity | Rank 1 | Rank 1 | Clean monotonic | Dominant, correct sign | All 4 agree — true primary driver |
| LED_Color | Activity | Rank 2 | Rank 2 | — | Secondary, correct sign | All agree — true secondary driver |
| Temperature_Rise_C | Activity | Rank 3 (near-zero) | Rank 3 (near-zero) | Flat/noisy | Near-zero | All 4 agree — decoy correctly ruled out |
| LED_Color | Selectivity | Rank 1 (42.4) | Rank 1 (1.93) | — | Dominant (-59) | All agree — true primary driver |
| Light_Intensity | Selectivity | Rank 3 | Rank 2 | Noisy, no trend | Small (+0.98) | Methods disagree on fine rank; all agree it's not a real driver |
| Temperature_Rise_C | Selectivity | Rank 2 | Rank 3 | Noisy, tail artifact | Small (+0.39) | Methods disagree on fine rank; all agree it's not a real driver |

**Key finding:** all four global/local methods correctly rule out the
temperature decoy as the dominant driver in both targets. They disagree on
its precise rank relative to the other weak, near-null feature — which is
itself informative: XAI methods are reliable at identifying strong signal
but can diverge in the low-signal regime, and reporting a single method's
output without cross-checking would risk overstating confidence in a small,
possibly spurious effect.

**Counterfactual highlight:** a self-written grid search, given a
low-selectivity instance (Red LED, 9.2%), found that switching to Green
LED alone -- with negligible change to intensity -- raises predicted
selectivity to 97.7%. This independently rediscovers the real published
finding (green LEDs -> ~100% mineralization selectivity) without being
told the mechanism.