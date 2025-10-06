# AeVAA - A Modular Psychologically-Grounded Valence-Aware Framework for Explainable ABSA
Part of the Quantum Criticism Research Project with: Assistant Professor David Guy Brizan, Alex Pezcon, and Harry Yin (MAGICS Lab @ University of San Francisco)

## Overview
A unified, graph-centric pipeline for tracking how sentiment attaches to and evolves around entities and their aspects within any text. ETSA processes raw sentences one by one, builds a dynamic property graph of entities, entity interactions, and descriptive phrases. The graph scores sentiment at multiple levels to generate a final, entity-centric sentiment.

## Setup

Create a virtual environment and install the Python dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

There may be a conflicting dependancy between maverick-coref and google-generativeai, however, it is resolvable by forcing protobuf to 3.20.* as google-generativeai does not need a later version of protobuf to operate properly.

## Running the Pipeline

The `pipeline` package exposes a CLI for caching datasets, running benchmarks, and executing ablation studies. Invoke it with Python's module flag from the project root:

```bash
python -m pipeline.main benchmark test_laptop_2014 --name demo_run
```

To pre-cache intermediate results prior to optimisation, run:

```bash
python -m pipeline.main cache test_laptop_2014
```

## Benchmark Outputs

Benchmark runs produce timestamped folders in `output/benchmarks/` containing metrics and graph snapshots for debugging. See the logs in each run directory for detailed insights.
