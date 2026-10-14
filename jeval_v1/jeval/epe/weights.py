from typing import Mapping

# Hardcoded risk weights by strata type for granular EPE decomposition and budget override.
RISK_WEIGHTS: Mapping[str, float] = {
    "FACTUAL": 1.0,
    "CAUSAL": 0.9,
    "ENTITY": 0.85,
    "TEMPORAL": 0.75,
    "CONTRASTIVE": 0.95,
    "BACKGROUND": 0.5,
}
