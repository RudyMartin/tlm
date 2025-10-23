# TLM Examples - Tensor Logic Math Primitives

This folder contains examples demonstrating the new Tensor Logic functions added to tlm.

## What's New

TLM has been enhanced with **Tensor Logic** support - temperature-controlled reasoning primitives based on Pedro Domingos's framework for unifying symbolic and analogical reasoning.

### New Modules

1. **tlm.core.similarity** - Advanced similarity and distance functions
2. **tlm.core.temperature** - Temperature scaling for reasoning control

## Examples

### 01_similarity_basics.py

Demonstrates core similarity operations:
- Cosine similarity between vectors
- Pairwise similarity matrices
- Top-K similar document retrieval
- Distance metrics (Euclidean, Manhattan, Cosine)
- Nearest neighbor search

**Run:**
```bash
cd examples
python 01_similarity_basics.py
```

### 02_temperature_control.py

Demonstrates temperature-controlled reasoning:
- Temperature-scaled softmax (deterministic vs exploratory)
- Temperature for reasoning modes (symbolic, hybrid, analogical)
- Temperature schedules for simulated annealing (linear, exponential, cosine)
- Temperature-controlled selection
- Practical document ranking example

**Run:**
```bash
cd examples
python 02_temperature_control.py
```

## Key Concepts

### Temperature Control

Temperature (T) controls the tradeoff between exploitation and exploration:

- **T ≈ 0**: Deterministic (symbolic reasoning, certifiable)
- **0 < T < 0.5**: Hybrid (mixed symbolic + analogical)
- **T ≥ 0.5**: Exploratory (analogical reasoning, similarity-based)

### Similarity vs Distance

- **Similarity**: Higher values = more similar (e.g., cosine similarity: 0-1)
- **Distance**: Lower values = more similar (e.g., Euclidean distance: 0+)

### Use Cases

- **Symbolic Reasoning**: Rule-based inference, formal verification (T=0)
- **Analogical Reasoning**: Case-based retrieval, similarity matching (T≥0.5)
- **Hybrid Reasoning**: Weighted combination of both approaches (0<T<0.5)
- **Simulated Annealing**: Optimization with temperature schedules
- **Neural Sampling**: LLM output control via temperature

## Integration

These primitives are part of the TidyLLM ecosystem:

1. **tlm** (this package) - Math primitives
2. **tidyllm-sentence** - Embedding + reasoning using tlm
3. **tidyllm** - Orchestration + temperature-controlled inference

See the main TLM README for full documentation.
