# JEval: JEPA-based Semantic Fidelity Compressor for Agent Memory

JEval is a standalone open-source library that compresses AI agent memory using Joint Embedding Predictive Architecture (JEPA) to maintain semantic fidelity while reducing storage costs.

## Features

- **JEPA-based Compression**: Uses predictive coding to compress semantically similar content
- **Artifact Protection**: Detects and protects critical artifacts (files, APIs, secrets) from compression
- **Adaptive Budgeting**: Allocates compression budget based on content importance (z-score calibration)
- **Pluggable Backends**: Extractive, LLM-based, and adaptive compression strategies
- **Deterministic Evaluation**: Comprehensive metrics for compression quality

## Installation

```bash
pip install jeval
```

Or from source:

```bash
git clone https://github.com/yourorg/jeval.git
cd jeval
pip install -e .
```

## Quick Start

```python
from jeval import JEval

# Initialize compressor
jeval = JEval()

# Compress a conversation
session = """
User: Fix the login bug in src/auth.ts
Agent: The issue is in the JWT validation. Here's the fix...
"""

compressed = jeval.compress_session(session, budget=0.5)
print(f"Compressed to {len(compressed)/len(session):.1%} of original")

# Evaluate fidelity
report = jeval.evaluate_compression(session, compressed)
print(f"Fidelity score: {report.fidelity:.3f}")
```

## Architecture

### Core Components

- **Encoders**: Frozen sentence transformer + trainable predictor head
- **EPE (Embedding Predictive Error)**: Measures semantic distortion
- **Strata Classification**: Content type detection (PROD/FAST)
- **Artifact Detection**: Regex-based identification of critical tokens
- **Compression Backends**: Multiple strategies with fallback
- **Evaluation**: Artifact recall, probe accuracy, compression ratios

### Pipeline

1. **Ingest**: Parse sessions into segments
2. **Encode**: Generate embeddings for all segments
3. **EPE Compute**: Calculate predictive errors
4. **Strata Classify**: Determine content importance
5. **Budget Allocate**: Z-score based budget distribution
6. **Compress**: Apply backend with artifact protection
7. **Evaluate**: Measure fidelity and effectiveness

## Configuration

JEval uses sensible defaults but can be configured:

```python
from jeval import JEval
from jeval.compress import AdaptiveCompressor

# Custom compressor
compressor = AdaptiveCompressor(
    encoder_model="all-mpnet-base-v2",
    predictor_layers=3,
    budget_threshold=0.8
)

jeval = JEval(compressor=compressor)
```

## Benchmarks

Compare against baselines:

```python
from jeval.baselines import BaselineFactory

# Test truncation baseline
baseline = BaselineFactory.create("truncation")
compressed = baseline.compress(text, budget=0.5)
```

## Training

Train custom predictor heads:

```python
from jeval.train import PairGenerator, train_predictor
from jeval.benchmarks import SWEBenchLoader

# Load training data
loader = SWEBenchLoader()
sessions = loader.load_sessions()

# Generate pairs
generator = PairGenerator(encoder)
pairs = generator.generate_pairs(sessions, num_pairs=10000)

# Train predictor
train_predictor(pairs, encoder, predictor, epochs=20)
```

## API Reference

### JEval

Main interface class.

- `compress_session(text, budget)`: Compress session text
- `evaluate_compression(original, compressed)`: Get fidelity report

### Compressors

- `AdaptiveCompressor`: Full JEPA pipeline
- `ExtractiveCompressor`: Sentence extraction
- `LLMCompressor`: OpenAI-based compression

### Evaluation

- `ArtifactEval`: Recall/F1 for protected artifacts
- `ProbeEvaluator`: LLM judge scoring
- `CompressionReport`: Aggregated metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{jeval2024,
  title={JEval: JEPA-based Agent Memory Compression},
  author={Preethi Shyam},
  year={2025},
  url={https://github.com/yourorg/jeval}
}
``` 
