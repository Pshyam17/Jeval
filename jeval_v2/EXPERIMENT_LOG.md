# Jeval v2 experiment log

Use one entry per hypothesis. **Result** must say whether it is a local synthetic check, a user-supplied simulation, a real-data measurement, or pending. No synthetic result may be described as a benchmark score. Dates use Pacific time.

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

## Result-entry template for future runs

**Hypothesis →**

**Setup and experiment →** Dataset/revision; split and source groups; compressor/prompt/budget; encoder/revision; score or verifier policy; selection criterion; seeds; exact commands and artifact hashes.

**Result →** Sample/group counts, point estimates and paired intervals, QA and evidence outcomes, calls/tokens/latency, failure breakdown, and missing data. State whether exploratory or locked test.

**Takeaways →** What the result supports, what it cannot support, decision to proceed/stop, and next hypothesis.
