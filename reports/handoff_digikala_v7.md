# Handoff v7: Digikala Review Project (Project 3). Replaces v3-v6.

## 0. Rules for the new assistant
- User speaks Persian: explain in Persian; keep code, names, paths in English.
- ONE step at a time, short separate Colab cells, say what each cell does. Never very long cells.
- Deadline: 2026-11-24 (tomorrow relative to 2026-11-23). Be fast, prioritize required deliverables.
- Leakage rules: model input = `body` only. Test set is evaluated ONCE per model; do NOT run any more test inference. Threshold 0.90 fixed. Pseudo reference labels only for post-hoc audit.
- Be honest in the report about every incident (section 5).

## 1. Status checklist (vs. course spec)
DONE
- Part A (A1-A5) over full data: scripts + JSON + figures (eda_utils.py, dedup.py, step_a1..a5.py, audit_verification.py). NOT YET: the notebook `01_descriptive_analysis.ipynb` that presents them.
- Part B: normalization, product-grouped 70/15/15 split (SPLIT_SEED 2026), masking 20% per class (MASK_SEED), split_manifest.json, subset zip.
- Part C: TF-IDF+LR baselines (both heads); fine-tuned xlm-roberta-base two-head model (supervised); semi-supervised experiment (pseudo-labeling, recommendation head) trained and tested.
REMAINING
1. Error analysis (>= 20 mistakes) [required].
2. Offline prediction script (normalize -> tokenize -> TF-IDF + transformer -> both heads, probabilities, short explanation; demo on short/ambiguous/negative/mixed reviews) [required].
3. Notebooks 01-04 + README + release_report.json + final report + presentation [required].
4. Backup of runs to laptop [urgent, Drive was unreliable].
Optional: A6 vocabulary analysis, threshold sweep, Phase 2 FastAPI/Docker.

## 2. Data/setup facts
Raw 6,156,289 rows; clean 6,152,858 (dedup: A=2911, B=113, C=202 duplicate ids, C dropped fully); 331,598 products; reviews per product capped ~200 by scraper (raw 400 = scraped twice). One out-of-range rate (2500, id 38631814). rate=0 rows suspicious (label noise); recommendation_status missing ~14.5%. Subset zip `digikala_subset.zip`: train_supervised_sub.csv (536,243), train_pseudo_unlabeled_sub.csv (110,744; id, body), pseudo_reference_labels_sub.csv (id, recommendation_status), validation_sub.csv (150,000), test.csv (922,953), manifests.
Model: xlm-roberta-base, mean pooling, two linear heads, MAX_LEN 128, batch 64, 3 epochs, lr_enc 2e-5, lr_head 5e-4, wd 0.01, warmup 6%, fp16, word embeddings frozen (277,457,670 params, 85,456,134 trainable), label smoothing 0.1, ordinal soft 0.2 for band, class weights, rate=0 rows masked in band loss, seed 2026. Library: `digikala_lib.py` (on Colab Drive MyDrive/digikala; user should upload it to the new chat).

## 3. Results (test, each evaluated once)
Supervised (runs/exp1_resume/best.pt = epoch 2; val rec 0.7042, band 0.5913):
| head/subset | n | F1 xlmr | F1 tfidf | balAcc | acc | MAE |
|---|---|---|---|---|---|---|
| rec all | 787720 | 0.7044 | 0.6601 | 0.7519 | 0.8278 | - |
| rec long_text | 583463 | 0.6997 | 0.6506 | 0.7503 | 0.8082 | - |
| rec text_not_in_train | 503212 | 0.6955 | 0.644 | 0.7459 | 0.7897 | - |
| band all | 922953 | 0.5918 | 0.5427 | 0.5909 | 0.6826 | 0.385 |
| band excl. rate0 | 844324 | 0.6644 | 0.5648 | 0.6953 | 0.735 | 0.281 |
| band long_text | 671222 | 0.5927 | 0.5338 | 0.5974 | 0.6633 | 0.405 |
| band text_not_in_train | 581146 | 0.5986 | 0.5312 | 0.6042 | 0.6597 | 0.409 |
Per class rec: recommended P.965 R.865 F1.913; no_idea P.375 R.599 F1.461; not_recommended P.694 R.791 F1.739. Band: low P.688 R.458 F1.550; medium P.353 R.530 F1.424; high P.820 R.785 F1.802.

Semi-supervised pseudo-labeling (threshold 0.90): pool 110,744; accepted 35,449 (32.01%): recommended 32,298 (91.11%), no_idea 78 (0.22%), not_recommended 3,073 (8.67%). Pseudo-label accuracy (post-hoc vs reference): 0.9865 overall; recommended 0.9926; no_idea 0.8590 (n=78); not_recommended 0.9261; rejected rows accuracy 0.7551 (threshold keeps easy rows).
Semi model: trained from pretrained xlm-roberta on train_supervised + accepted pseudo rows (571,692 rows), pseudo rows only in recommendation loss (band target -100), same settings/seed. Validation: epoch0 rec .6745/band .5682; epoch1 .7074/.5974 (best, score .6524); epoch2 .7057/.5862 (score .6459). IMPORTANT: eval_test was run on the END-OF-TRAINING model (epoch 2), not best.pt (epoch 1). Disclose this; do not re-run test.
Semi test (end-of-training model): rec Macro F1 0.7047 (supervised 0.7044), acc 0.8239 (0.8278), balAcc 0.7519 (0.7519); rec long_text F1 0.6999, text_not_in_train 0.6958; no_idea P.369 R.629 F1.465; not_recommended P.713 R.768 F1.739; recommended P.967 R.859 F1.910. Band (not the target of this experiment): all F1 0.5866 (0.5918), MAE 0.395 (0.385), excl. rate0 0.6602 (0.6644).
Conclusion: no meaningful gain (+0.0003 F1, within noise, single seed). Reasons: pseudo rows only ~6.6% extra, 91% majority class, easy rows (confirmation bias), only 78 no_idea. Band difference is not attributable to pseudo-labeling (band head gets no pseudo loss).
Files on Drive (account 2, MyDrive/digikala/runs/): exp1_resume/{best.pt, test_preds.npz, test_metrics.json, history.json, tokenizer, pseudo_preds.npz}; semi/{best.pt (epoch 1), last.pt, test_preds.npz, test_metrics.json, history.json}.

## 4. NEXT STEPS (detailed)
1. DOWNLOAD runs/exp1_resume and runs/semi (without last.pt) plus digikala_lib.py; empty Drive trash; Colab: Runtime -> Disconnect and delete runtime.
2. Error analysis (laptop, no GPU): load test.csv + test_preds.npz of both models (arrays id, pr, pb). Pick >= 20 errors across categories: negation, sarcasm, short text, mixed opinion, spelling variation, duplicated text, contradictory labels, band-boundary cases (rate near 2.5/4.0), rate=0 noise, TF-IDF vs transformer disagreement. Use a fixed seed for sampling and document it. Also discuss weak classes (no_idea, medium).
3. Offline predict script (src/predict.py): needs saved TF-IDF vectorizer+LR (from 02_baselines; verify they were saved), tokenizer in runs/exp1_resume/tokenizer, best.pt weights (trainable params only; load into TwoHead with strict=False after loading pretrained xlm-roberta-base). Works on CPU.
4. Notebooks: 01 (from artifacts/eda JSONs and figures), 02 (TF-IDF metrics json), 03 (training details, history.json curves, params, supervised test table), 04 (semi results above).
5. README, release_report.json, final report (include incidents), slides.

## 5. Incidents to disclose honestly
1. Account-1 Drive filled (old ~1 GB last.pt versions in trash), final epoch weights were not saved there; stale best.pt (epoch 0) and last.pt (step 15,000).
2. A first test evaluation used the stale epoch-0 checkpoint (rec 0.675, band 0.545); discarded. Model selection is by validation; the final supervised test was run once with the recovered epoch-2 model.
3. Training resumed from step 15,000 on account 2; resumed validation (0.7042/0.5913) matches the original history (0.7039/0.5914).
4. GPU quota of account 1 ended; Kaggle phone verification failed; Colab account 2 (T4) used. Semi training was interrupted by a runtime disconnect and resumed automatically/from checkpoint (ckpt every 3000 steps).
5. Semi model test evaluation used the end-of-training model, not the validation-best epoch (see section 3).
6. Transformer trained on a documented subset (train 536,243) due to hardware limits; test set is full (922,953).
