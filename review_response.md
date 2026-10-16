# Response to the Review

This document maps each point of the review of the paper to the changes made. All numbers are from the rerun experiments (generated automatically into `latex/numbers.tex`).
For both LR and HGB, the results of the existing methods are identical to those before the revision (2,430 rows each, difference 0).
Section and table numbers are the same in the English (`latex/main.pdf`) and Japanese (`latex/main_ja.pdf`) versions of the paper.

## Weaknesses

### W1 / Issue 3: Missing CleanLearning baseline
- Added `cl_cleanlearning` (`cleanlab.classification.CleanLearning`) (Section 3.3 (2), Tables 1–3). Because it receives the same `pred_probs`, it removes exactly the same samples as `cl_prune`; we verified and state that the only difference is the per-class reweighting 1/max(T̂_kk, 1e-3).
- Result: CleanLearning did not outperform `cl_prune` (LR −0.75±0.22 pt, win rate 17%; HGB −0.06±0.17 pt, win rate 37%). The gap to `bq/naive/irbl` actually widened (LR +1.20, HGB +0.37 pt). The hypothesis that "reweighting narrows the gap" was not supported (Section 6, RQ1).
- Section 3.3 now states explicitly that `cl_prune` is pruning with uniform weights.
- The limitations note that CleanLearning's training time is not reported because it includes re-running the issue detection internally and is not comparable.

### W2: Confounding between trusted-set size and noise concentration in U
- Added `oracle_trusted30/irbl` and `oracle_check --oracle-frac 0.3` with a genuine trusted set of 30% (Tables 1, 2, 5). The 10% and 30% sets come from the same random draws and are nested. Section 3.3 states that sampling is uniform without stratification, that all remaining samples form U, and why 10% and 30% were chosen.
- With matched sizes, human-verified IRBL beats `bq/naive/irbl` by LR +3.91 pt (win rate 72%) and HGB +5.00 pt (win rate 92%) (Section 5.3). HGB `trusted_only` reaches +6.81 with 30% oracle data, so the reversal at 10% (−0.16) is explained by the trusted-set size (Section 5.5).
- Filled the naive `trusted_only` cell of Table 5 with the `cl_prune` value, marked with †.
- Showed that IRBL's gain over T-only training is larger for random (+0.58 / +1.91) than for naive (+0.45 / +0.31), and argued that part of naive's advantage comes from the size of T (Sections 5.2 and 6, RQ1).
- Section 3.4 gives the range of naive |T| (LR 53–85%, HGB 53–83%), and Sections 3.4 and 5.4 note that the error rate of U for random/topconf exceeds ρ.

### W3: Related work
- Added George et al. (TMLR 2024, arXiv:2410.15772) in a paragraph on detecting mislabeled examples, positioning this study as evaluating detectors by their usefulness as input to biquality learning rather than by detection accuracy.
- Added Swayamdipta et al. (EMNLP 2020, Dataset Cartography) in a paragraph on the selection bias of confidence-based filtering, and referred to it in the discussion of RQ2.

### W4: Preprocessing and generation details
- Stated that `topconf` and `cl_feature` use self-confidence (Sections 3.2–3.4).
- Added the formula for normalized margin, (p_ỹ − max_{k≠ỹ} p_k + 1)/2.
- Stated that s in `cl_feature` is appended unstandardized in [0,1] and that s=1 at inference is the upper end of its range.
- Listed all parameters of `make_classification`.

## Potential Issues

### Section 2: gold estimation in transition-matrix methods
- Section 2 now explains that bqlearn adapts Plugin and IRLNL to the biquality setting with the same gold estimator as GLC (a classifier trained on U evaluated on T), whereas the original methods estimate T from noisy data, e.g. via anchor points. This explains why all three break down together.

### Numbering in the abstract
- Rewrote the sentence without numbers.

### Standardization, instance noise, and synth
- Stated that standardization is fitted once on the 70% training split before CV and applied to all 54 features of covtype, including the 44 binary ones.
- Stated that f* is an LR (max_iter=2000) trained on the true labels of the 70% training split, that its probabilities on the training data are used, and that selection is weighted sampling without replacement.
- Section 4 now also mentions the asymmetry that noise generated with LR concentrates near the LR decision boundary.

### Issue 4: HGB learning rate and internal estimators
- Disclosed that the learning rate of 0.05 was originally chosen by looking at the test data of seed 0.
- Checked it post hoc with 5-fold CV within the training data, without test data (`hgb_lr_check`, Table 11 in the appendix). On covtype, 0.1 gives 76.1% (minimum 72.1%) and 0.05 gives 79.1%, so the same choice is made. On digits and synth the difference was at most 0.4 pt.
- Stated that the internal classifiers of IRBL and the gold estimator are the base model itself, and that K-PDR uses LR for density-ratio estimation regardless of the base model.
- Added to the limitations that the selection process depended on the test data.

### Omission of naive/trusted_only
- Section 3.3 and the note to Table 1 state that it is omitted because it is identical to `cl_prune`.

### Mechanism analysis
- Extended to all 90 conditions and both models (previously LR, seed 0 only). Removed the corresponding limitation.
- Table 7: GLC estimates are now compared with the clean rate of U in addition to the true value over D, and oracle10/30 columns were added. With naive, the clean rate of U is only 0.35 (LR), and the estimate of 0.22 falls even below that.
- New finding: GLC underestimates the diagonal even with genuine trusted data (LR 0.50 and HGB 0.62 vs. a clean rate of 0.70 in U), and this does not change at 30%. We present, as a hypothesis, the interpretation that the classifier's own error is added on top of the noise. We also state that for HGB the estimate is close to the true value but the accuracy drop is large, so the bias of the estimate alone cannot explain it.
- Table 8: Added IRBL weights for random and topconf. The ratio of weights on correct vs. wrong labels is naive > random > topconf, supporting the explanation for RQ2.
- Showed that at LR instance 40% the weight on wrong labels is 0.80 (mean over all seeds; 0.98 for seed 0) and barely lower than for correct labels, which explains why `bq/naive/irbl` falls below `cl_prune` in this condition.
- Tested the minority-class hypothesis with Table 6 (oracle breakdown by dataset). The degradation of Plugin is concentrated on covtype, consistent with the hypothesis, but GLC and IRLNL also degrade for LR on the class-balanced synth, which minority classes alone cannot explain.

### Tables
- Added a row with the absolute baseline accuracy to Tables 3, 9, and 10.
- The note to Table 2 states that differences are computed from unrounded values.
- Added random U to Table 4, and the note states that naive U equals the precision and naive |T| = 1 − the flagged fraction.
- The note to Table 7 states that the true value is the unweighted average over classes, (1/K)Σ T_kk, and does not match ρ under instance noise.

### Appendix B
- Grouped the commands in a single block after a `\clearpage`, including `--oracle-frac 0.1/0.3`, `mechanism --model`, `hgb_lr_check`, and `wsl sh latex/build.sh`. Stated that `export_latex` generates Figure 1, all tables, and the number macros in the text, and that everything can be run with `rerun_all.sh`.

### Discussion of cl_feature (Section 6, RQ3)
- Added the following three points, each marked as unverified:
  1. The shift in the distribution of s between training and inference (s=1) adds a constant offset to each class's intercept. The strong degradation of LR even under symmetric noise is consistent with this.
  2. This limitation is specific to additive concatenation without interaction terms.
  3. For HGB, splits on s act as implicit pruning inside the trees.

### Dataset dependence (Sections 6 RQ2 and 7)
- Added to the discussion and the limitations that for LR on covtype, every split that narrows the trusted set to 30% falls below the baseline (random_irbl −0.53, random_T −1.14; the only positive gains are naive_irbl +0.53 and cl_prune +0.14), whereas for LR on digits topconf_irbl (+7.01) outperforms random_irbl (+5.41).

## Effect on the conclusions
- The main conclusions hold: IRBL × naive is the best automatic method, transition-matrix methods break down, and topconf is worse than random.
- We weakened the claim that "IRBL × naive is better than pruning", attributing part of the benefit to the size of the trusted set.
- We quantified the "cost of removing humans entirely" with a size-matched comparison (30%).
