# BARDIC

**Base-level Agent for Retrieval, Discourse, and Internal Consistency**

> **Development status: active research / unstable**
>
> BARDIC is a work in progress. The repository currently contains an
> intermediate implementation of the project, not the complete architecture
> described in the roadmap below.
>
> Recent post-course development has made the current branch unstable, and a
> fresh clone should **not** be assumed to run end to end without additional
> work.

BARDIC is an experimental agentic AI system for **retrieval-grounded creative
text generation**.

The project began as a graduate-level investigation into whether several
specialized transformer-based components could cooperate to retrieve context,
generate structured poetry, evaluate that output, and eventually improve
generation through critic-derived feedback.

The intended mature design is a **three-transformer architecture** with distinct
roles for:

1. context and retrieval,
2. creative generation,
3. learned criticism.

That complete architecture is **not implemented yet**.

The current repository contains working pieces of that design, experimental
scaffolding around others, and some repository drift caused by continued
development after the original course submission.

---

## What BARDIC Is Right Now

The current implementation contains substantial work toward:

- Wikipedia-based contextual retrieval
- semantic anchor extraction
- sentence-transformer embeddings
- query and context vector generation
- T5 / FLAN-T5-based candidate poetry generation
- candidate filtering and fallback recovery
- hybrid poem evaluation
- structured critic scores and diagnostics
- JSONL experiment logging
- Docker and Docker Compose scaffolding
- infrastructure intended to support future critic-driven training experiments

At present, the implemented system uses **two transformer model families**.

### `all-MiniLM-L6-v2`

Used through Sentence Transformers for:

- query embeddings
- contextual embeddings
- semantic-anchor ranking
- semantic relevance measurements
- some of the current ArtCritic's evaluation metrics

### T5 / FLAN-T5 family

Used for:

- candidate creative-text generation
- the current `models/bardic_base` experimental generator

The current `ArtCritic` is **transformer-backed**, because it uses
transformer-derived embeddings.

It is **not a third independent learned transformer model**.

That distinction is important.

---

## What BARDIC Is Not Yet

BARDIC should **not currently be described as a completed three-transformer
system**.

The public repository does not yet contain the dedicated learned critic
transformer envisioned in the original architecture.

The current `ArtCritic` is a hybrid evaluation system. It combines
transformer-derived semantic similarity with deterministic scoring logic for
qualities including:

- four-line structure
- instruction cleanliness
- cadence
- query alignment
- retrieved-context alignment
- fragment and list-like output detection
- repetition
- poetic texture
- rhyme

This evaluator is useful engineering scaffolding.

It is **not the final ArtCritic architecture**.

BARDIC is also **not currently a production-ready application**.

The project has not yet reached the reproducibility, reliability, automated
testing, packaging, deployment, or output consistency expected of a production
system.

The current public branch is under active reconstruction and may not execute
successfully from a clean clone.

---

## Why BARDIC Exists

BARDIC explores a deceptively difficult question:

> Can several small, specialized transformer-based models cooperate to produce
> creative text that is grounded, structured, evaluated, and iteratively
> improved without requiring one enormous general-purpose language model to do
> everything?

Poetry was chosen partly because creative text exposes failure modes very
quickly.

A generated poem may be grammatical while still failing because it:

- ignores the requested subject
- echoes its own instructions
- collapses into fragments
- repeats words or line openings
- violates structural constraints
- misses cadence or rhyme requirements
- produces generic prose rather than poetic language
- terminates after one or two tokens
- produces no useful output at all

These behaviors make poetry a useful test bed for:

- constrained creative generation
- small-model behavior
- model orchestration
- semantic grounding
- guardrail design
- learned evaluation
- feedback-driven training

---

# Current Architecture

The current implementation is best understood as an **incomplete research
pipeline**, not the final BARDIC architecture.

```text
User Query
    |
    v
Wikipedia Retrieval
    |
    v
RAGEncoder
    |
    +-- MiniLM embeddings
    +-- semantic anchor selection
    +-- query/context vectors
    |
    v
T5Poet
    |
    +-- T5-family generator
    +-- candidate generation
    +-- filtering / fallback recovery
    |
    v
[repair / style stage currently under reconstruction]
    |
    v
Current ArtCritic
    |
    +-- MiniLM semantic similarity
    +-- deterministic quality checks
    |
    v
Scores + Diagnostics + JSONL Logs
