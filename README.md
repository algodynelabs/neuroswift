# NeuroSwift

**Real-Time Neuro-Symbolic Grounding via Asynchronous VLM Supervision**

This repository contains the implementation for the paper:

> *Real-Time Neuro-Symbolic Grounding via Asynchronous VLM Supervision*
> Brad Lishman, Algodyne Labs

## Overview

NeuroSwift is a dual-process architecture combining fast temporal-difference learning (>1000Hz on CPU) with sparse VLM queries triggered only when novel visual features are recruited. The VLM initializes feature weights based on semantic valence (e.g., "fire" → negative), which the RL agent then refines through experience.

### Key Results

- **107× query reduction** compared to frame-by-frame VLM integration
- **0% first-episode death rate** (vs 40% for uninformed agents) when encountering fire
- **Hallucination robustness**: incorrect VLM priors produce no measurable performance degradation

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For VLM experiments, install Ollama and pull a model
# https://ollama.ai
ollama pull llava
```

## Project Structure

```
neuroswift/
├── src/
│   ├── agents/           # SwiftTD and Q-learning agents
│   ├── environments/     # TextureGrid environment
│   ├── vlm/              # VLM oracle and trigger mechanisms
│   ├── utils/            # Tracking utilities
│   └── visualization/    # Plotting functions
├── experiments/          # Experiment scripts (A, B, C)
├── tests/                # Unit tests
├── data/
│   └── textures/         # Tile textures for environments
└── docs/
    └── paper.tex         # LaTeX source for the paper
```

## Running Experiments

### Experiment A: Semantic Jump

Tests whether VLM-initialized weights help avoid fire on the first episode.

```bash
python -m experiments.experiment_a --seeds 10 --episodes 100
```

### Experiment B: Hallucination Robustness

Tests whether incorrect VLM priors are corrected through TD learning.

```bash
python -m experiments.experiment_b --seeds 10 --episodes 100
```

### Experiment C: Trigger Efficiency

Compares query counts across trigger mechanisms (Frame, Imprint, Random, None).

```bash
python -m experiments.experiment_c_full_statistical --seeds 10 --episodes 100
```

## Running Tests

```bash
pytest tests/ -v
```

## Architecture

```
┌─────────────────┐
│   Environment   │
│  (TextureGrid)  │
└────────┬────────┘
         │ obs
         ▼
┌─────────────────┐
│    Feature      │
│   Extraction    │
└───┬─────────┬───┘
    │         │
    ▼         ▼
┌───────┐ ┌───────────┐
│System1│ │  System2  │
│SwiftTD│◀│ VLM Oracle│
│>1000Hz│ │  ~0.5Hz   │
└───┬───┘ └───────────┘
    │ action
    ▼
┌─────────────────┐
│   Environment   │
└─────────────────┘
```

## Citation

If you use this code, please cite:

```bibtex
@article{lishman2025neuroswift,
  title={Real-Time Neuro-Symbolic Grounding via Asynchronous VLM Supervision},
  author={Lishman, Brad},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
