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
python -m jeval_v2.experiment embed --pairs /path/to/pairs.jsonl --output outputs/pairs.npz
python -m jeval_v2.experiment diagnose --embeddings outputs/pairs.npz --split validation
python -m jeval_v2.experiment train --embeddings outputs/pairs.npz --output checkpoints/predictor.pt
python -m jeval_v2.experiment evaluate --embeddings outputs/pairs.npz --checkpoint checkpoints/predictor.pt --split test --output outputs/test_scores.jsonl
```

The test score file includes cosine, predictive residual, identity residual, and the fraction of whitespace-delimited words removed for the **same rows**. The latter is a cheap length baseline, not a tokenizer-specific compression ratio. Evaluation prints the mean cosine between the prediction and compressed embedding. Summary metrics print only when both harm classes have labels. Rejection fractions are descriptive and do not choose thresholds on test. The embedding step rejects text beyond the encoder token limit. Training checkpoints include the embedding file hash and encoder name, and evaluation requires an exact match. For a paper, compute grouped bootstrap confidence intervals and train an equally supervised classifier baseline on the same external training data; this scaffold has neither yet.

The label-free `diagnose` gate fits a ridge-regularized linear correction from compressed to original embeddings on training groups, then compares squared error with identity on held-out groups. It reports mean cosine to the compressed input and a group-bootstrap interval for error reduction. Run on validation before model work; reserve test for a final check. A reconstruction gain establishes a learnable shift, not better harm detection. Near-identity cosine alone is not a stop rule. Fix ridge using validation only and record the choice.

## Northeastern Explorer

Northeastern moved public GPUs from Discovery to **Explorer**. Verify your assigned Slurm account and partitions with `sinfo` and `sacctmgr show assoc user=$USER` after logging in. Submit the example scripts in `jeval_v2/slurm` after replacing the example paths. Install dependencies and download the encoder in a compute session; do not run training on the login node. Put caches and outputs in your assigned scratch or project storage.
