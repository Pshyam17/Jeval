# Jeval v2 experiment log

Use one entry per hypothesis. **Result** must say whether it is a local synthetic check, a user-supplied simulation, a real-data measurement, or pending. No synthetic result may be described as a benchmark score. Dates use Pacific time.

## E00 — Headline paired LoCoMo answer experiment (revised 2026-10-01; pending)

**Hypothesis →** Given the same LoCoMo question, compressed memory, prompt, and answer model, an EPE-derived risk percentile leads to more correct/evidence-supported answers than a cosine-derived percentile.

**Setup and experiment →** Freeze encoder, compressor, EPE checkpoint, LoCoMo release, question list, prompt template, model version, and decoding settings. Construct two prompts per question from the same memory. The only differing field is the 0–100 within-run midrank risk percentile; randomize A/B assignment and store the key separately. Run both through the same LLM; fill the blinded review sheet against official answers and evidence before opening the key. Primary descriptive comparison is paired counts (EPE-only correct, cosine-only correct, both, neither), with conversation-level breakdown over LoCoMo's ten source conversations. Report answer support and abstention separately. Do not infer missing facts from the risk score.

**Result →** **Pending.** No compressed LoCoMo memories, score-joined question rows, LLM answers, or human judgments exist. `locomo_prompt_pairs.py` renders paired prompts and a review sheet, but does not call a model or calculate EPE.

**Takeaways →** This directly tests the user's intended LLM-input comparison. It cannot claim better retrieval or source recovery. If both risk percentiles are nearly identical or the model ignores the field, the experiment may be null; report that rather than modifying the prompt after viewing test answers.

## E01 — Identity EPE is cosine (2026-09-28)

**Hypothesis →** Identity prediction adds no ranking signal to cosine on unit embeddings.

**Setup and experiment →** For normalized `X,C`, compare `||X-C||²` with `2(1-X·C)`. `tests/test_core.py` asserts equality for simple vectors.

**Result →** Exact algebraic equivalence; direct identity-control test is present but full pytest was unavailable locally. The user also reported correlation 1.000 in a simulation. No dataset result.

**Takeaways →** Identity EPE is a control, not a competing method; improvement requires learned ranking changes.

## E02 — The original bottleneck can manufacture divergence (2026-09-28)

**Hypothesis →** A 768→512→768 path cannot represent identity and may change EPE without learning useful lost information.

**Setup and experiment →** User simulated unbottlenecked, bottlenecked, and residual predictor classes. V2 added `C + MLP(C)` and a test that a zeroed MLP exactly represents identity even with a narrow hidden layer.

**Result →** **User simulation, not Jeval measurement:** reported EPE/cosine correlation ~0.88 for the bottleneck and ~0.999 with a residual correction. Local syntax checks passed; full PyTorch test unrun.

**Takeaways →** Use residual predictor and identity diagnostics. Divergence alone does not demonstrate harm detection.

## E03 — A systematic compressor transform can make EPE useful (2026-09-28)

**Hypothesis →** A learned predictor can remove a repeatable compressor shift, leaving unusual information loss as residual error.

**Setup and experiment →** User constructed an invertible synthetic transform with an additional harmful loss, plus a control with only unstructured severity changes.

**Result →** **User simulation/existence proof:** systematic case cosine AUROC 0.805, EPE AUROC 1.000; unstructured case both AUROC 1.000 with score correlation ~0.998. Not run on natural language pairs.

**Takeaways →** Test learnability of the compressor's transform on independent real pairs before labeling benchmark harm. Perfect synthetic AUROC is not a forecast.

## E04 — Heteroscedastic NLL (rejected, 2026-09-28)

**Hypothesis →** Predicting variance with the mean might improve harm ranking.

**Setup and experiment →** User compared variance-normalized NLL with plain residual in a simulated harmful-loss regime.

**Result →** **User simulation:** residual AUROC 1.000 versus NLL 0.598. Not implemented in v2.

**Takeaways →** Variance can normalize away the signal; omit this arm from the bounded study.

## E05 — Linear diagnostic gate (local synthetic checks, 2026-09-28)

**Hypothesis →** Matrix correction generalizes beyond a group-weighted intercept-only shift.

**Setup and experiment →** Fit ridge on train groups, report held-out identity, intercept, and matrix squared errors; group-bootstrap the **intercept-minus-matrix** reduction. Synthetic shifted unit vectors and identity data were tested directly in the local NumPy environment. `tests/test_diagnose.py` also encodes these checks for pytest.

**Result →** **Local synthetic control only:** on one seeded 800-row constant-shift-like construction, identity error 0.053397, intercept error 0.000948, matrix error 0.000906; incremental group reduction ~0.000042 with CI [0.0000093, 0.0000766]. This is tiny compared with identity-to-matrix gain ~0.05249, showing why identity-to-matrix was the wrong gate. The matrix gains a little because normalization makes a nominally constant shift slightly nonlinear. Identity-input control gave essentially zero improvement. Real compressor data: **pending**.

**Takeaways →** Report the incremental effect and its size, not just a CI excluding zero. Predeclare what magnitude warrants downstream investment after observing independent development data, before the locked benchmark.

## E06 — Training and detector comparison (pending)

**Hypothesis →** On real fixed-compressor pairs, MSE-selected EPE ranks question-relevant harm better than cosine and clipped word removal.

**Setup and experiment →** Proposed source: MSC train conversations, disjoint by conversation; frozen compressor, budget, and encoder. Train EPE with no harm labels in the objective or headline checkpoint selection; evaluate on held-out LongMemEval-S cleaned question/evidence instances, with LoCoMo secondary. Report AUPRC, AUROC, fixed-budget recall, paired group intervals, and seeds 0–2. Logistic classifier and AUPRC-selected EPE are separately supervised comparisons.

**Result →** **Pending:** MSC pairs, compressor, benchmark-side pairs, validated question-relative harm labels, trained checkpoints, and score files do not exist yet.

**Takeaways →** Do not claim EPE beats cosine. A pair-level result alone cannot establish answer repair.

## E07 — Hired/fired and other state reversals (pending)

**Hypothesis →** Pair-only scores may miss a later reversal; a question-aware source retrieval trigger can repair it.

**Setup and experiment →** Develop matched cases with hired→fired, fired→rehired, retained reversal, and equally long irrelevant deletions. Use two questions against the same compression (e.g., “when hired?” versus “currently employed?”). On the locked LongMemEval test, stratify by knowledge updates and temporal reasoning; use evidence-session and turn metadata, and distinguish compression omission from retrieval and answer failures.

**Result →** **Pending:** no matched natural-language set, labels, retrieval policy, answerer run, or benchmark score.

**Takeaways →** Harm is `(source, compression, question)`-relative. A question-blind detector has an intrinsic ceiling.

## E08 — Surprisal fallback at equal verifier cost (pending)

**Hypothesis →** A frozen-LM score on omitted source text adds useful triage signal beyond EPE, or a question-aware retrieval trigger beats both.

**Setup and experiment →** Align source and compressed text; score omitted spans by token NLL conditioned on preceding source context. Freeze model, tokenizer, prompt, and score aggregation. On independent development data set trigger thresholds or EPE+surprisal fusion. On the locked benchmark compare no verification, random, length, cosine, EPE, surprisal, EPE+surprisal, query-trigger, and always-verify ceiling using the same raw-turn retriever and answerer at matched 5/10/20% call budgets. Measure source-turn recall, repaired answers, calls, tokens, and latency.

**Result →** **Pending:** surprisal scorer, alignment, verifier, and end-to-end QA are not implemented. No cost or accuracy result.

**Takeaways →** Surprisal is not a proof of temporal relevance. Keep it only if it improves answer accuracy per unit cost over EPE and a simple query-based trigger.

## E09 — Benchmark manifest adapter (implemented; no benchmark evaluation)

**Hypothesis →** Official question/evidence metadata can be normalized without manufacturing harm labels or leaking benchmark data into training.

**Setup and experiment →** `python -m jeval_v2.benchmarks.prepare --dataset longmemeval|locomo --input OFFICIAL.json --output MANIFEST.jsonl`. LongMemEval adapter preserves question type, answer session IDs, and marked evidence turn indexes. LoCoMo adapter preserves question category, evidence dialogue IDs, and source conversation group. Tiny fixture tests check the mapping.

**Result →** **Implemented, awaiting full dataset execution.** No official dataset downloaded or committed, and no harm label generated. Fixture tests are present but pytest was unavailable in the local environment.

**Takeaways →** Manifests identify evaluation questions and evidence. Compression, counterfactual labels, retrieval, and QA judging remain separate work.

## E10 — Reproducibility checks (2026-09-28)

**Hypothesis →** Code and cluster entrypoints parse and synthetic controls show intended mechanics.

**Setup and experiment →** Python compilation, shell syntax, diff whitespace checks; direct NumPy diagnostic and cross-seed aggregation scripts. The Slurm script schedules a CPU array with seeds 0–2.

**Result →** **Local checks passed** on existing v2 code through PR commit `c9f7becc` before this log. The full torch/pytest suite and any HPC or real benchmark run remain **pending**.

**Takeaways →** Passing syntax and synthetic controls is only implementation verification. Update this entry with exact commands, hashes, dataset revision, seed files, and measured outputs after the first real run.

## E11 — LoCoMo best-of-N: cosine versus EPE candidate selection (real-data run, 2026-10-14; exploratory, single seed)

**Hypothesis →** With the same frozen compressor and the same pool of sampled compressions per segment, choosing the candidate with the lowest EPE yields better downstream LoCoMo answers than choosing the one with the lowest cosine distance, at matched length. This is a *selection* experiment. It is **not** E00 (score shown to the answer model) and **not** E06 (harm-label detection); both remain pending.

**Setup and experiment →**
- *Data.* Official `locomo10.json` (SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`), ten conversations. Cut into 1095 segments of 6 turns each (session date header and `[shared image: caption]` kept; longest segment 379 words, 479 encoder tokens, 0 over the 512 limit). QA: categories 1–4 only, **1540 questions**; category 5 (adversarial, no gold answer) skipped.
- *Compressor.* `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28`, vLLM 0.31.0, torch 2.13.0+cu129, one H200. One fixed prompt (sha256 `f4bec9f3…3f32`), target 0.5 of the original words, N=8 samples per segment, temperature 0.8, top-p 0.95, seed 0. 8760 candidates, none empty. The ratio, N, temperature and band tolerance were fixed before the run and not tuned afterwards.
- *Scores.* Encoder `BAAI/bge-base-en-v1.5` @ `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. Cosine is `1 − cos(original, candidate)`. EPE is the squared residual of the residual-MLP predictor (`C + MLP(C)`, hidden 1536) trained to predict the original embedding from the compressed one, label-free, checkpoint chosen by validation MSE, seed 0. 5-fold cross-fitting over whole conversations (6 train / 2 validation / 2 test); each conversation's candidates were scored by a predictor that never trained on that conversation, and the test folds cover all ten conversations exactly once.
- *Selection arms.* Per segment, among candidates within ±25% of the word budget (whole pool if none, 149 segments): `cosine` (lowest distance), `epe` (lowest residual), `random` (seed 0), `first` (lowest candidate index), plus an uncompressed `original` memory as the reader's ceiling.
- *Reader.* The same Qwen model, greedy decoding, the whole assembled memory in the prompt, short-phrase answers. Metric: token F1 following the LoCoMo conventions for categories 1 and 3, no stemming. Paired difference with a bootstrap over the ten conversations (1000 draws, seed 0).
- *Jobs and files.* Slurm: compress 11071799 (16 min), folds 11071800 (array 0–4, 1.5–3 min each), QA 11071801 (6.5 min). Outputs under `/scratch/shyamsundar.p/runs/locomo-r50-s0/`: `segments.jsonl`, `candidates.jsonl` (+ metadata), `fold-*/scores.jsonl`, `selected/`, `predictions.jsonl`, `report.json`. Code at commit `0c07cb9` (the first launch, job 11071627, failed on an optional `torchcodec` import that needed CUDA 13 libraries; that package was removed and nothing else changed).

**Result →** **Real-data measurement, one seed, exploratory; no locked test.**

| Arm | Mean F1 | Words kept / original |
|---|---|---|
| original (uncompressed) | 0.3781 | 1.000 |
| EPE | 0.3416 | 0.492 |
| cosine | 0.3384 | 0.497 |
| first | 0.3369 | 0.464 |
| random | 0.3363 | 0.469 |

F1 by category (1 / 2 / 3 / 4): original 0.286 / 0.149 / 0.151 / 0.522; EPE 0.287 / 0.168 / 0.138 / 0.450; cosine 0.279 / 0.154 / 0.125 / 0.453; first 0.271 / 0.162 / 0.115 / 0.451; random 0.295 / 0.158 / 0.125 / 0.442.

Paired mean-F1 differences with 95% bootstrap intervals over the ten conversations:

| Comparison | Difference | 95% interval |
|---|---|---|
| EPE − cosine | +0.0031 | [−0.0083, +0.0119] |
| EPE − random | +0.0052 | [−0.0080, +0.0153] |
| cosine − random | +0.0021 | [−0.0109, +0.0163] |
| EPE − first | +0.0046 | [−0.0094, +0.0178] |

Selection diagnostics: cosine and EPE picked the **same candidate for 82.4% of segments**, leaving about 190 segments where the arms differ. 149 of 1095 segments (13.6%) had no candidate in the length band and used the whole pool. The gap from the best compressed arm to the uncompressed memory is 0.037 F1 (EPE) and 0.040 (cosine).

*Not measured or not yet read:* the predictor diagnostics (`mean_prediction_cosine_to_compressed` and `mean_absolute_epe_minus_identity` in each fold's `evaluate` output and the train logs) were not examined; harm labels do not exist, so no AUPRC/AUROC; no second or third training seed; no evidence-restricted analysis; no LLM judge; no correction for the four paired comparisons (none is individually significant).

**Takeaways →**
- **No detectable difference between EPE and cosine selection.** The point estimate favors EPE by 0.3 F1 points, but its interval contains zero, as do those of the comparisons with random and first. Per-category differences (EPE ahead by 1.4 points on categories 2 and 3, slightly behind on category 4) are post hoc, without intervals, and should not be read as findings.
- **This is not evidence that the scores are equivalent.** The intervals rule out an EPE advantage much larger than about one F1 point on this setup, but the setup has low power: all four compressed arms lie within 0.6 points of each other, against a 3.7-point drop from the uncompressed memory, and selection recovered almost none of it.
- **Dilution is the first suspect.** With 82% identical choices, only the remaining segments can move the two arms apart, and most questions do not depend on them. A test restricted to questions whose evidence lies in segments where the choices differ would be more sensitive, but would be post hoc and must be labelled secondary.
- **Length control held for the central comparison** (EPE 0.492 vs cosine 0.497 words kept). `first` and `random` kept about three points less text, so their comparison with the scored arms leans slightly toward the scores; this does not affect EPE-versus-cosine.
- **Protocol deviations to keep in view.** (1) The predictor was trained on LoCoMo's own conversations (cross-fitted, label-free, QA labels never used), which departs from the rule in the README and research log that LoCoMo must not train or select the predictor; the proposed independent MSC source was not used. A result from this setup cannot count as a locked benchmark result. (2) The compressor and the reader are the same model. (3) Token F1 is a noisy proxy.
- **Pending follow-ups, none run:** read the predictor diagnostics; evidence-restricted analysis of differing segments; predictor seeds 0–2 with the existing aggregator; a harder setting (lower ratio, more and more varied candidates); an independent training source and LongMemEval for more independent groups.

## Result-entry template for future runs

**Hypothesis →**

**Setup and experiment →** Dataset/revision; split and source groups; compressor/prompt/budget; encoder/revision; score or verifier policy; selection criterion; seeds; exact commands and artifact hashes.

**Result →** Sample/group counts, point estimates and paired intervals, QA and evidence outcomes, calls/tokens/latency, failure breakdown, and missing data. State whether exploratory or locked test.

**Takeaways →** What the result supports, what it cannot support, decision to proceed/stop, and next hypothesis.
