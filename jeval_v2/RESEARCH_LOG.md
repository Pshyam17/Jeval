# Jeval v2 research log

## Current central hypothesis (revised 2026-10-01, Pacific time)

On the same LoCoMo questions, with identical compressed memory, prompt wording, and answer model, providing an EPE-derived compression-risk percentile may produce more correct and evidence-supported answers than providing a cosine-derived percentile. Answers are compared in paired, blinded manual review. **Only the score varies between the two arms.** This tests the *usefulness of the score as model input*, not whether EPE improves retrieval, recovers deleted facts, or outperforms a full memory system. No result exists yet.

The score cannot supply omitted evidence. On a hired→fired compression that only says “hired,” neither arm can know the later event; a useful signal may instead increase appropriate abstention. Compare both correctness and support, and record cases where the scores lead to identical prompts/answers. Use midrank percentiles to put the two score families on a common 0–100 scale; this is a presentation choice fixed before the test.

## Earlier broader hypothesis (2026-09-28, superseded as the headline)

For a **fixed compressor and budget**, the compression may impose a reproducible transform on source embeddings. A predictor trained on independent original–compression pairs could learn this transform. Its residual may then rank **question-relevant information loss** better than cosine distance, at equal encoder and data budgets. This only helps if (1) the transform generalizes beyond an intercept-only shift, (2) the encoder represents the lost state, and (3) the residual correlates with downstream harm. Pair-only scores cannot know which future question will be asked. A question-conditioned source retrieval fallback might repair missed state updates; surprisal is an optional triage signal that must earn its extra cost.

**Earlier proposed falsification:** on held-out groups, a learned matrix would improve reconstruction over an intercept-only correction; an integrated verification system would improve accuracy per call, token, and millisecond. These remain exploratory diagnostics or future work. They are no longer the headline LoCoMo experiment. No real-data result is available yet.

## Scope and provenance

**2026-10-01 correction:** The user narrowed the experiment to two parallel LoCoMo runs of the same answer model: one receives cosine risk and one EPE risk, with manual paired comparison. The earlier LongMemEval-first and verifier-policy designs below are historical proposals and should not be read as the active protocol. See [`LOCOMO_PROTOCOL.md`](LOCOMO_PROTOCOL.md).

- `jeval_v1/` archives the original repository; its older `RESEARCH_LOG.md` records prior code work and is not evidence for v2.
- `jeval_v2/` is a fresh detector scaffold. Its dataset JSONL is an input contract; no training corpus, compressor, encoder, harm labels, or benchmark answers have been produced by v2.
- Draft [PR #2](https://github.com/Pshyam17/Jeval/pull/2) holds this work. This log reflects the branch state as of 2026-09-28 PT.
- Statements below marked **user simulation** were supplied during audit, not reproduced on benchmark data by Jeval.

## Decisions and reasoning

### Research boundary

The paper target is one compressor at one budget, one primary encoder with a prespecified ablation, cosine versus trained prediction residual, and a fixed-cost retrieval policy. The headline experiment uses no harm labels to train or select EPE: its validation checkpoint criterion is MSE. AUPRC checkpoint selection and the logistic classifier are separately labeled-supervision comparisons. We do not claim JEPA, a general memory architecture, or state-of-the-art memory QA from pair-level detector results.

### Mathematical audit and correction

For unit vectors, identity prediction gives `||X-C||² = 2(1-X·C)`. Thus its AUROC/AUPRC ranking is cosine's ranking. The initial 768→512→768 predictor could not represent identity through its bottleneck, potentially creating a spurious difference. We added a residual path `C + MLP(C)`, made hidden width 1536 by default, removed hardcoded weight decay, and report mean `cos(pred,C)`. **User simulation:** identity gave correlation 1.000; a bottleneck gave about 0.88, while a residual model approached 0.999. The synthetic numbers are not empirical results on our corpus.

**User simulation:** an invertible systematic compressor transform plus idiosyncratic harmful loss gave cosine AUROC 0.805 and EPE AUROC 1.000 in a clean-room construction. Unstructured loss yielded near-identical rankings. These examples motivate the mechanism; they do not forecast real performance. The proposed heteroscedastic/NLL score was retracted after a user simulation where it suppressed the harmful residual magnitude.

### Diagnostic gate

`diagnose.py` fits a group-weighted ridge correction on train pairs and evaluates identity, an intercept-only shift, and the full matrix on disjoint held-out groups. The **incremental intercept-to-matrix error reduction**, with a group-bootstrap interval, is the deciding statistic. Identity-to-matrix improvement alone can be explained by a constant shift. Ridge values may be compared on validation only; one preselected value is permitted on test. A reconstruction gain is necessary for the proposed transform account but cannot establish question harm detection. Synthetic shift and identity controls were executed locally; see `EXPERIMENT_LOG.md`.

### Embedding, training, and metrics

The encoder model and revision are explicit arguments. `inspect-lengths` reports token coverage and `embed` rejects silent truncation. Embedding files include source IDs, group IDs, split, optional harm, and word ratio; a sidecar records encoder provenance. Evaluation checks checkpoint/embedding hash and handles legacy checkpoints with a retrain error. Multiple seeds use separate files. Evaluation reports cosine, EPE, identity, clipped words-removed, a separately supervised logistic baseline, AUPRC/AUROC, fixed rejection fractions, grouped bootstrap intervals, and paired EPE-minus-cosine intervals. `aggregate.py` reports between-seed mean, standard deviation, and range on identical held-out IDs.

The score `harm` is **question-relative**: a compression can retain the answer to “When was Maya hired?” and lose the answer to “Does Maya currently work there?” The current pair-only predictor does not take a question. We must use question IDs and source evidence in the benchmark protocol; the detector's ceiling should be discussed explicitly.

### Benchmarks and training source

**Selected primary held-out target:** [LongMemEval-S cleaned](https://github.com/xiaowu0162/longmemeval), with predeclared `knowledge-update` and `temporal-reasoning` slices and full-set reporting. It includes timestamped histories, question types, answer session IDs, and marked evidence turns. This is more directly suited to stale “hired → fired” state than agent task trajectories. [Official schema](https://github.com/xiaowu0162/longmemeval#-dataset-format).

**Secondary held-out target:** [LoCoMo](https://github.com/snap-research/locomo), the ten released conversations with annotated QA and evidence dialogue IDs. It supports a conversation-memory comparison, including future MemRefine-style evaluation; ten conversation groups limit uncertainty estimation. [Official data](https://github.com/snap-research/locomo#data).

**Proposed independent training source:** [Multi-Session Chat](https://parl.ai/projects/msc/) train conversations. A fixed compressor would create `(original, compressed)` pairs, with disjoint source conversations for train/validation. This is a proposal, not acquired data. We must pin corpus revision, compressor model/prompt, token budget, segmentation, and encoder before running. Neither LongMemEval nor LoCoMo may select the model, threshold, or fusion weights for a locked benchmark result. [MSC release](https://parl.ai/projects/msc/).

`benchmarks/prepare.py` now creates read-only question/evidence manifests from official local files; it does **not** download or commit benchmark data, create compressions, infer harm, run retrieval, or answer questions. LongMemEval question IDs do not necessarily identify independent source histories; group overlapping evidence histories before making inferential claims. LoCoMo groups questions by the conversation `sample_id`.

AMA-Bench remains a possible later transfer test, not the selected primary hired/fired benchmark. Its published dataset is test-only, and end-to-end integration with its two-stage memory interface is not implemented. [Official repository](https://github.com/AMA-Bench/AMA-Bench).

### Surprisal and source verification

The proposed first surprisal arm is frozen-LM negative log likelihood of **omitted source content conditioned on preceding source context**. It is a write-time triage score, requires source/compression alignment, and cannot independently decide which omission matters to a future question. This score is not implemented. At query time, a fixed raw-source retriever plus the same answerer can be called under equal budgets for random, length, cosine, EPE, surprisal, EPE+surprisal, and question-based verification policies. Report evidence retrieval and answer repair separately. This is a protocol proposal, not an experiment already run.

### Cluster and reproducibility

Example Slurm scripts require an explicit repository root, assigned account/partition at submission, and scratch/project model cache. Embedding requests one GPU; the small predictor uses a three-seed CPU array. `python -m compileall`, `bash -n`, `git diff --check`, direct synthetic gate checks, and direct seed aggregation checks passed in the local workspace. Full PyTorch/pytest suite did not run here because those packages were unavailable. No Explorer job has been submitted.

## Outstanding decisions before a benchmark result

1. Pin exact MSC release and source-group construction, fixed compressor and prompt, target token budget, segmentation, and encoder revision after length audit.
2. Define an auditable question-conditioned harm protocol: original evidence correct, compressed evidence wrong/unsupported, plus a fixed answerer and judge; categorize omission, retrieval, and answer failures separately.
3. Implement pair creation, benchmark-side compression and answerer integration, source retrieval, surprisal alignment/model, cost accounting, and verifier policies. Freeze all thresholds and fusion weights on independent development data.
4. Check for overlap among LongMemEval histories before group bootstrap; do not treat 500 questions as 500 independent sources without checking.
5. Run the real-pair diagnostic first. Stop the EPE transform claim if matrix gain over intercept does not generalize; even a positive gain is not a harm-detection result.

## Chronology of v2 repository work (2026-09-28 PT)

1. Archived the old code as `jeval_v1/`; scaffolded v2 detector, data contract, predictor, CLI, tests, and Slurm examples in draft PR #2.
2. Audited score equivalence, bottleneck, truncation, provenance, label granularity, selection, and missing baselines. Added residual prediction, identity/length controls, encoder length checks, provenance, and diagnostic output.
3. Added a label-free ridge reconstruction gate, then corrected its deciding comparison from identity-to-matrix to **intercept-to-matrix** after an audit found the first statistic rewarded a constant shift.
4. Required explicit encoder revision; added grouped metrics, paired bootstrap, a separately supervised logistic comparison, MSE headline selection, explicit AUPRC ablation, CPU seed array, and cross-seed aggregator.
5. Chose LongMemEval-S cleaned as the proposed primary stale-state evaluation and LoCoMo as secondary. Added benchmark manifest adapters and this log. No corpus, compressor run, model training, real benchmark labels, or QA evaluation has occurred.
