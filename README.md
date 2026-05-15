# BARDIC

**Base-level Agent for Retrieval, Discourse, and Internal Consistency**

BARDIC is an experimental agentic AI poetry system that retrieves contextual information from Wikipedia, extracts semantic anchors, generates a short poem from those anchors, repairs the output with a style/rhyme layer, and evaluates the result with a transformer-backed critic.

The project was built as an independent graduate-level AI system exploring retrieval-augmented generation, structured creative text generation, semantic scoring, rhyme repair, and critic-driven feedback for future reinforcement-learning workflows.

---

## Overview

BARDIC takes a user-provided topic or query and runs it through a multi-stage creative generation pipeline:

1. **Retrieve context** from Wikipedia.
2. **Extract semantic anchors** from the retrieved article text.
3. **Generate a structured poem** using a T5-based model.
4. **Repair and stylize the poem** with a Spin Doctor layer.
5. **Evaluate the poem** using a transformer-backed critic.
6. **Log critic results** for debugging, analysis, and future reinforcement-learning experiments.

The goal is not simply to generate text. BARDIC is designed as a structured creative AI pipeline where generated output can be grounded, shaped, repaired, evaluated, and improved.

---

## Features

- Retrieval-augmented generation using Wikipedia as a knowledge source
- Semantic anchor extraction using sentence embeddings and article-wide keyword selection
- T5-based poem generation with candidate filtering and fallback recovery
- Spin Doctor repair layer for style smoothing and ABAB rhyme repair
- Mythic-name phonetic normalization for rhyme handling
- Transformer-backed critic using sentence-transformer similarity scoring
- Multi-factor poem evaluation, including:
  - structure
  - instruction cleanliness
  - cadence
  - context alignment
  - query alignment
  - list-fragment control
  - repetition control
  - poetic texture
  - rhyme
- JSONL logging of generated poems, critic feedback, embeddings, and scores
- Docker-based environment setup
- Configuration-driven model and generation settings
- Designed with future reinforcement-learning feedback loops in mind

---

## Tech Stack

- **Python 3.12**
- **PyTorch**
- **Transformers**
- **Sentence Transformers**
- **T5 / FLAN-T5**
- **Wikipedia API**
- **NLTK**
- **pronouncing**
- **Docker**
- **Docker Compose**
- **YAML configuration**
- **JSONL logging**

---

## Repository Structure

```text
BARDIC/
├── config/
│   └── settings.yaml
├── core/
│   ├── art_critic.py
│   ├── evaluation.py
│   ├── logger.py
│   ├── mythic_phonetics.py
│   ├── rag_encoder.py
│   ├── spin_doctor.py
│   ├── t5_poet.py
│   └── test_poems.txt
├── logs/
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
└── BARDIC_ Base-level Agent for Retrieval,Discourse, and Internal Consistency.pdf
```

---

## Core Components

### `main.py`

Runs the interactive BARDIC pipeline:

- loads configuration from `config/settings.yaml`
- initializes the RAG encoder
- initializes the T5 poem generator
- initializes the Spin Doctor repair layer
- initializes the critic
- accepts a user query
- retrieves Wikipedia context
- generates a draft poem
- repairs and stylizes the poem
- evaluates the final poem
- logs the run to `logs/critic_runs.jsonl`
- optionally triggers reinforcement-learning training when enough critic logs exist

---

### `core/rag_encoder.py`

The retrieval and semantic-anchor module.

Responsibilities include:

- querying Wikipedia
- selecting a relevant page
- extracting a lead summary or fallback summary
- filtering poor article chunks
- extracting noun-based candidate keywords
- ranking candidates with sentence-transformer embeddings
- selecting semantically useful keyword anchors
- returning summary text, anchors, and embeddings for downstream generation and evaluation

---

### `core/t5_poet.py`

The structured poem generator.

Responsibilities include:

- loading a T5-compatible model from the configured model path
- building prompt seeds from semantic anchors
- generating candidate poem lines
- cleaning model output
- filtering instruction echo and malformed text
- scoring lines by syllable count, lexical variety, poetic diction, and semantic similarity
- assembling a four-line poem from candidate line banks
- falling back to template lines only if generation cannot produce usable output
- exposing `compute_logprob()` for future reinforcement-learning training

---

### `core/spin_doctor.py`

The poem repair and style layer.

Responsibilities include:

- preserving fallback markers so the critic can penalize fallback behavior
- normalizing poem length to four lines
- applying light stylistic modes, including minimal, lyrical, archaic, and modern transformations
- extracting line-ending words
- checking rhyme relationships
- repairing ABAB rhyme when possible
- using safe fallback rhyme pairs when phonetic lookup fails

The Spin Doctor is intentionally light-touch: the Poet writes the structure, and the Spin Doctor gives the output a final polish.

---

### `core/mythic_phonetics.py`

A phonetic normalization helper used by the Spin Doctor.

This module maps difficult or mythic names into rhyme-friendly phonetic forms so the rhyme system has a better chance of handling uncommon names and nonstandard spellings.

---

### `core/art_critic.py`

The transformer-backed evaluation module.

The critic evaluates each generated poem with multiple subscores:

- **Structure:** checks for a four-line poem
- **Instruction cleanliness:** penalizes prompt echo or fallback markers
- **Cadence:** checks syllable range per line
- **Context vector alignment:** compares poem embedding to retrieved context embedding
- **Query vector alignment:** compares poem embedding to query embedding
- **List-fragment control:** penalizes fragmentary or list-like output
- **Repetition control:** penalizes excessive repeated words or repeated openings
- **Poetic texture:** compares the poem against poetic and plain-prose reference embeddings
- **Rhyme:** checks an ABAB-style rhyme pattern

The critic returns a scalar score, a 10-point score, a status label, subscore details, and human-readable feedback.

---

## Configuration

The main settings live in:

```text
config/settings.yaml
```

The configuration includes:

- project metadata
- RAG source and embedding model
- model paths
- generation temperature and top-p sampling
- maximum source and target lengths
- number of generation attempts
- poem constraints
- reinforcement-learning training settings
- critic thresholds

Example configuration areas:

```yaml
rag_encoder:
  source: "wikipedia"
  embedding_model: "all-MiniLM-L6-v2"

poet_decoder:
  model_name: "models/bardic_base"
  max_source_length: 256
  max_target_length: 128
  max_new_tokens: 80
  temperature: 0.95
  top_p: 0.92
  num_attempts: 24

bardic_constraints:
  output_type: "quatrain"
  lines_per_stanza: 4
  target_syllables_per_line: 10
```

---

## Installation

### Option 1: Docker Compose

Clone the repository:

```bash
git clone https://github.com/Lacynth40/BARDIC.git
cd BARDIC
```

Create a `.env` file if you use a Hugging Face token:

```bash
HF_TOKEN=your_huggingface_token_here
```

Build the Docker image:

```bash
docker compose build
```

Run the interactive program:

```bash
docker compose run --rm bardic-agent
```

---

### Option 2: Local Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main.py
```

---

## Usage

When the program starts, it prompts for a topic or query:

```text
=== BARDIC STRUCTURED POET ===

Enter topic/query:
```

Example input:

```text
The Battle of Thermopylae
```

BARDIC then:

1. searches Wikipedia
2. selects a page
3. extracts a summary
4. extracts semantic anchors
5. generates a draft poem
6. repairs/stylizes the poem
7. evaluates the poem
8. logs the run

Example output structure:

```text
[RAG] Fetching context...

[Summary]
...

[Keywords]
['Sparta', 'Persian', 'Battle', 'King']

[Generating]

[Draft]
...

[Spin Doctor]
...

[Final]
...

[Critic]
Scalar score: ...
Score / 10: ...
Status: ...

Subscores / 10:
  structure: ...
  instruction_cleanliness: ...
  cadence: ...
  context_vector_alignment: ...
  query_vector_alignment: ...
  list_fragment_control: ...
  repetition_control: ...
  poetic_texture: ...
  rhyme: ...
```

---

## Logging

BARDIC logs each critic run to:

```text
logs/critic_runs.jsonl
```

Each log entry may include:

- query
- retrieved summary
- semantic anchors
- draft poem
- final poem
- critic score
- critic status
- critic subscores
- critic feedback
- query embedding
- context embedding

These logs are intended for debugging, evaluation, and future reinforcement-learning experiments.

---

## Current Status

BARDIC is an experimental research and portfolio project, not a production package.

The system demonstrates:

- retrieval from Wikipedia
- semantic keyword extraction
- T5-based poem generation
- rhyme/style repair
- transformer-backed scoring
- structured logging
- Dockerized execution

Known development areas:

- improve model output consistency
- strengthen setup documentation for local model paths
- expand automated tests
- improve dependency pinning and environment portability
- refine reinforcement-learning training integration
- add sample generated outputs to the repository
- add screenshots or architecture diagrams
- document GPU/CPU expectations more clearly

---

## Notes on Model Files

The configuration expects a local model path:

```text
models/bardic_base
```

If this directory is not present, the model loader may fail unless the configuration is changed to point to an available Hugging Face model or a locally downloaded model directory.

The Dockerfile pre-caches:

- `google/flan-t5-large`
- `all-MiniLM-L6-v2`

If you want to run BARDIC directly from the pre-cached FLAN-T5 model instead of a local model directory, update `config/settings.yaml` accordingly.

---

## Research Context

This project was developed as part of graduate-level AI experimentation around structured creative generation.

BARDIC explores how generated poetry can be:

- grounded in retrieved factual context
- shaped by semantic anchors
- repaired for style and rhyme
- filtered for structural constraints
- evaluated with transformer embeddings
- logged for possible reinforcement-learning feedback

The repositor
