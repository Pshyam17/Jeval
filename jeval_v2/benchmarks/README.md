# Held-out benchmark manifests

The selected **primary** target is [LoCoMo](https://github.com/snap-research/locomo#data): paired runs of the same answer model with the same compressed memory, varying only whether the prompt contains cosine or EPE risk. [LongMemEval-S cleaned](https://github.com/xiaowu0162/longmemeval#-data) remains an optional later transfer benchmark; it is not part of the headline experiment. No benchmark data is bundled or used for EPE training. See the [LoCoMo protocol](../LOCOMO_PROTOCOL.md).

Download/pin the official releases locally, then create question/evidence manifests:

```bash
python -m jeval_v2.benchmarks.prepare --dataset longmemeval --input /data/longmemeval_s_cleaned.json --output outputs/longmemeval_questions.jsonl
python -m jeval_v2.benchmarks.prepare --dataset locomo --input /data/locomo10.json --output outputs/locomo_questions.jsonl
```

The adapter does **not** compress text, retrieve evidence, infer harm, or run QA. Its output must never be passed directly as the `(original, compressed)` training JSONL. LongMemEval uses question ID as a manifest identifier, not as proof that histories across questions are independent; check shared source sessions before group-based uncertainty estimates. LoCoMo uses `sample_id` as the conversation group. Preserve the raw official dataset revision/hash alongside any later compression and answer artifacts.

Training is proposed on independent [Multi-Session Chat](https://parl.ai/projects/msc/) conversations after one compressor, budget, and encoder are frozen. The exact pair generator and question-relative harm protocol remain unimplemented. See [the research log](../RESEARCH_LOG.md) and [the experiment log](../EXPERIMENT_LOG.md).
