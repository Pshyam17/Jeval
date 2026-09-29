# Jeval v2: cosine versus trained predictive error

## Scope

One frozen encoder and one frozen compressor produce an original segment `x` and candidate memory `c`. The cosine score is `1 - cos(f(x), f(c))`. The predictor is trained only on `train` pairs to estimate the normalized original embedding from the normalized compressed embedding. Its test score is squared prediction residual. No test-set labels or conversations are used to train it.

This is a detector experiment, **not yet a memory-system benchmark**. A downstream gate and QA counterfactual evaluation must be implemented before claiming improved answer accuracy. No result is claimed by this scaffold. An identity predictor has residual exactly twice cosine distance on unit embeddings, so only a learned change in ranking could support the hypothesis. The residual predictor starts with an identity skip path; its output cosine to the compressed embedding is reported to expose a near-identity solution.

## Input

JSONL, one candidate per row:

```json
{"id":"unique-pair-1","group_id":"source-conversation-1","split":"train","original":"The meeting moved to Tuesday.","compressed":"The meeting moved.","harm":1}
```

`id`, `group_id`, `split`, `original`, and `compressed` are required. `harm` is optional and must be `0` or `1` when supplied. It is an **evaluation label**: a source-supported future question becomes unanswerable or wrong from the compressed evidence while remaining answerable from the original. Do not treat a synthetic deletion as proof of harm. The reader rejects a `group_id` crossing splits.

This definition of harm depends on the future question. The current pair-only scores do not see that question; a compression can receive different labels for different questions. For a question-conditioned study, retain a question ID in the source manifest, sample each question explicitly, and add a question-aware baseline. Do not present pair-only detector metrics as an answer to the question-conditioned task.

Create pairs using a **single fixed compressor and target budget**. Do not use LoCoMo or LongMemEval histories to train or select the predictor. Keep a separate manifest recording dataset revision, original conversation ID, compressor model/prompt/version, target tokens, and evidence question ID. The generator and the QA labeling protocol are intentionally not hidden inside this detector package: those research choices must be fixed before results are reported.

## Run

From the repository root, in a Python 3.10+ environment:

```bash
python -m pip install -r jeval_v2/requirements.txt
python -m jeval_v2.experiment inspect-lengths --pairs /path/to/pairs.jsonl --encoder MODEL_ID --encoder-revision COMMIT_SHA
python -m jeval_v2.experiment embed --pairs /path/to/pairs.jsonl --encoder MODEL_ID --encoder-revision COMMIT_SHA --output outputs/pairs.npz
python -m jeval_v2.experiment diagnose --embeddings outputs/pairs.npz --split validation --ridge 0.0001 0.001 0.01
python -m jeval_v2.experiment train --embeddings outputs/pairs.npz --output checkpoints/predictor-0.pt --seed 0
python -m jeval_v2.experiment evaluate --embeddings outputs/pairs.npz --checkpoint checkpoints/predictor-0.pt --split test --output outputs/test_scores-0.jsonl
python -m jeval_v2.aggregate outputs/test_scores-{0,1,2}.jsonl
```

Choose the encoder after `inspect-lengths` reports its actual token coverage. No encoder is assumed. Pin a specific model revision, produce separate embeddings and checkpoints per encoder, and keep source pairs and splits fixed for an encoder ablation. The test score file includes cosine, predictive residual, identity residual, and a nonnegative fraction of whitespace-delimited words removed (with raw word ratio for expansions). Evaluation reports grouped bootstrap intervals and the paired EPE-minus-cosine interval. A logistic classifier trained on labeled training embeddings `[x; c; x-c]` is also scored when both training classes exist. It uses standardized features and fixed `C=1.0`; with 3× embedding dimensions this can overfit and is a different-budget comparison, not a tuned upper bound. The embed step rejects text beyond the encoder limit. Old checkpoints without provenance require retraining.

Headline checkpoint selection uses validation MSE and no harm labels. `--selection auprc` is a separately reported label-tuned ablation that requires both validation harm classes; it must not be pooled with the headline runs. The selection criterion and chosen epoch are recorded. Run predeclared seeds 0, 1, and 2 with separate checkpoint and score paths; do not select a seed on test. `jeval_v2.aggregate` checks that the score files contain identical held-out pairs and reports mean, sample standard deviation, and range over seeds. Group bootstrap intervals are within-seed and do not measure across-seed uncertainty.

The label-free `diagnose` gate fits a ridge-regularized linear correction on training groups. It compares identity, an intercept-only shift, and the full matrix on held-out groups. **The deciding statistic is intercept error minus matrix error, with a group-bootstrap interval.** A full map that only beats identity but not the intercept has not established the matrix mechanism. Sweep ridge on validation only; test accepts one preselected value. A reconstruction gain establishes a learnable shift, not better harm detection. Near-identity cosine alone is not a stop rule.

## Northeastern Explorer

Verify your assigned Slurm account and partition on the cluster before submission. Export absolute `JEVAL_REPO_ROOT`, `JEVAL_PYTHON`, paths, and `JEVAL_CACHE_DIR` on scratch/project storage; embed also needs `JEVAL_ENCODER` and `JEVAL_ENCODER_REVISION`. The CPU training script runs seeds 0–2 as an array and needs `JEVAL_CHECKPOINT_DIR`. Submit with `sbatch --account="${JEVAL_ACCOUNT}" --partition="${JEVAL_GPU_PARTITION}" jeval_v2/slurm/embed.sbatch`; submit the CPU training job with `sbatch --account="${JEVAL_ACCOUNT}" --partition="${JEVAL_CPU_PARTITION}" jeval_v2/slurm/train.sbatch`. Export variables using `sbatch --export=ALL` if your cluster does not propagate them by default. The scripts do not assume an account or CPU partition. Encoder cache is under `JEVAL_CACHE_DIR`; install dependencies and download models from a compute session, and keep output outside home quota.
