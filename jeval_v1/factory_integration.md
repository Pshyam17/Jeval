# Factory Droid Integration Plan for Jeval v3.0

This document outlines the strategy for deeply integrating Jeval's architecture with the Factory AI Droid ecosystem, allowing us to collect real-world coding data and actively augment the Droid's memory.

## Phase 1: Passive Data Collection (Implemented)

**Goal:** Build a massive, organic dataset of real-world coding trajectories to benchmark Jeval against.

### 1. PreCompact Hook (Active)
- **What it does:** Captures the session history exactly when the context window fills up, right before Factory Droid compresses it.
- **Implementation:** Configured in `.factory/settings.json`.
- **Status:** Done. Script located at `jeval/benchmarks/capture_droid_hook.py`.

### 2. SessionEnd Hook (Next Step)
- **What it does:** Captures the full history of every completed session, even if it didn't hit the token limit. This provides a baseline of successful tasks without context loss.
- **Implementation Plan:** 
  Append the following to `.factory/settings.json`'s `hooks` array:
  ```json
  {
    "event": "SessionEnd",
    "matcher": ".*",
    "command": "python3 /Users/preethi/jevalV2/Jeval/jeval/benchmarks/capture_droid_hook.py"
  }
  ```

## Phase 2: Active Data Tagging

**Goal:** Make it easier to generate automated DroidBench probes (RECALL, ARTIFACT, DECISION questions) from the captured data.

### 1. `AGENTS.md` Directives
- **What it does:** Forces the Code Droid to explicitly tag its reasoning and artifacts in the chat, making them trivial to parse later.
- **Implementation Plan:**
  Create an `AGENTS.md` file in the root of the project being worked on with the following instruction:
  ```markdown
  # Jeval Data Collection Directives
  When making architectural choices or resolving complex bugs, explicitly state your reasoning using the format:
  `DECISION: [Choice] BECAUSE [Reasoning]`
  When creating or modifying critical files, state:
  `ARTIFACT: [File Path]`
  ```

## Phase 3: Interactive Tooling

**Goal:** Allow the user to interact with Jeval directly from the VS Code Droid interface.

### 1. Custom Slash Command (`/jeval`)
- **What it does:** A custom Droid command that snapshots the current conversation, runs it through the Jeval v3.0 pipeline, and outputs compression stats (token reduction, fidelity score) back into the chat.
- **Implementation Plan:**
  Create `.factory/commands/jeval.sh` (or equivalent configuration) that triggers a lightweight CLI wrapper around `JevalMemory`.

## Phase 4: Full Active Memory (MCP Server)

**Goal:** Replace or supplement Factory Droid's native memory with Jeval's fidelity-gated architecture.

### 1. Model Context Protocol (MCP) Integration
- **What it does:** Exposes Jeval's `JevalMemory` as a local MCP server.
- **Implementation Plan:**
  - Wrap `JevalMemory` in a fast local API (e.g., using FastAPI, similar to the demo server).
  - Define MCP tools: `store_memory(text)`, `retrieve_memory(query)`.
  - Configure `.factory/settings.json` to connect to this local MCP server.
  - The Droid can now actively query Jeval when it needs to recall older context that has been summarized away by the native system.
