#!/usr/bin/env python3
"""
Example usage of jeval for compressing agent memory.
"""

from jeval import JEval

# Initialize JEval
jeval = JEval()

# Sample session
session_text = """
User: I need to implement user authentication.
Agent: Use JWT tokens. Store in Redis.
User: How to handle expired tokens?
Agent: Check expiry and refresh.
"""

# Compress
compressed = jeval.compress_session(session_text, budget=0.5)
print("Compressed:", compressed)

# Evaluate
report = jeval.evaluate_compression(session_text, compressed)
print("Fidelity:", report.fidelity)