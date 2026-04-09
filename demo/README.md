# Jeval Live Demo

Two-panel research showcase demonstrating pre-hoc semantic fidelity
evaluation for agent memory compression in real time.

## Quick start

    pip install -r demo/requirements.txt
    python3.12 demo/run_demo.py

Opens http://localhost:8765 automatically.

## Controls

**Left panel — Latent Space**
- Rotate: mouse drag
- Zoom: scroll wheel
- Click any point: inspect original and compressed text
- Type a query in the bottom input: watch relevant points highlight

**Right panel — Compression Theater**
- Automatic — shows each compression event as it happens
- Gold tokens: anchor tokens preserved verbatim
- Cyan tokens: paraphrased content
- Purple tokens: LLM-generated tokens not in original

## Speed control

    python3.12 demo/run_demo.py --speed 0.5   # half speed, good for walkthroughs
    python3.12 demo/run_demo.py --speed 2.0   # double speed

## What to watch for

1. **Redundancy rejection (entries 8-9):** second identical ambient
   entry appears as a hollow ring in Panel 1 and shows
   "NOVELTY GATE: REJECTED" in Panel 2

2. **Contradiction detection (entries 15 and 17):** deployment failure
   followed by deployment success — red line appears between the two
   points in Panel 1, old point dims to gray

3. **Precision retrieval:** type "what happened at step 11" in the
   query box — the relevant points highlight and the fact index
   returns the exact step content

4. **Anchor preservation:** watch for gold tokens in Panel 2
   on entries containing JWT_SECRET, step references, error codes

## Pre-demo checklist

Run through this before presenting. Do not skip steps.

1. Start server 2 minutes before presenting:
       python3.12 demo/run_demo.py
   Wait for "NIM warm — opening browser" before touching anything else.
   NIM cold-starts take 30-90 seconds on first call after inactivity.

2. Verify both panels are visible and the scene is ready (not yet
   replaying — browser connected, waiting for you to signal start).
   The replay starts automatically 2 seconds after browser connects.

3. Close all other browser tabs on the demo machine to prevent
   accidental WebSocket connections starting the replay early.

4. Have this query ready to paste into the retrieval box for the
   audience interaction moment:
       what step did the JWT fix happen at

5. After the showcase, retrieve the query log:
       open http://localhost:8765/queries

## Replay speed

    python3.12 demo/run_demo.py --speed 0.7   # slower for walkthroughs (~6 min)
    python3.12 demo/run_demo.py --speed 1.0   # normal speed (~4 min)
    python3.12 demo/run_demo.py --speed 2.0   # fast for time-limited demos (~2 min)

## What to narrate at each moment

- seq=9 cold_only: "The novelty gate rejected this — it is semantically
  identical to the previous entry. Only written to cold storage."

- seq=11, seq=13 cold_only: "These step observations are too similar to
  earlier cached content to warrant a new cache entry."

- contradiction event: "The system detected that the deployment outcome
  contradicts an earlier entry. The stale point dims in the latent space."

- audience retrieval: "Type any question about the session. Watch which
  channel answers it — precision queries hit the fact index directly."
