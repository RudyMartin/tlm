"""
Example 2: Temperature Control for Reasoning

Demonstrates temperature scaling functions from tlm.core.temperature
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tlm

print("=" * 60)
print("TLM Temperature Control - Examples")
print("=" * 60)

# Example 1: Temperature-Scaled Softmax
print("\n1. Temperature-Scaled Softmax")
print("-" * 40)
logits = [2.0, 1.0, 0.5]

# Low temperature (deterministic)
probs_cold = tlm.temperature_scaled_softmax(logits, temperature=0.01)
print(f"Logits: {logits}")
print(f"T=0.01 (deterministic): {[f'{p:.3f}' for p in probs_cold]}")

# Standard temperature
probs_normal = tlm.temperature_scaled_softmax(logits, temperature=1.0)
print(f"T=1.0  (standard):      {[f'{p:.3f}' for p in probs_normal]}")

# High temperature (uniform/exploratory)
probs_hot = tlm.temperature_scaled_softmax(logits, temperature=10.0)
print(f"T=10.0 (exploratory):   {[f'{p:.3f}' for p in probs_hot]}")

# Example 2: Temperature for Reasoning Control
print("\n2. Temperature for Reasoning Modes")
print("-" * 40)
similarity_scores = [0.9, 0.7, 0.5, 0.3]

print(f"Original scores: {similarity_scores}")

# Symbolic reasoning (T=0): deterministic
scaled_symbolic = tlm.apply_temperature(similarity_scores, temperature=0.0001)
print(f"T≈0 (symbolic):  {[f'{s:.3f}' for s in scaled_symbolic]}")

# Hybrid reasoning (T=0.3): moderate scaling
scaled_hybrid = tlm.apply_temperature(similarity_scores, temperature=0.3)
print(f"T=0.3 (hybrid):  {[f'{s:.3f}' for s in scaled_hybrid]}")

# Analogical reasoning (T=2.0): exploratory
scaled_analogical = tlm.apply_temperature(similarity_scores, temperature=2.0)
print(f"T=2.0 (analog):  {[f'{s:.3f}' for s in scaled_analogical]}")

# Example 3: Temperature Schedules (Simulated Annealing)
print("\n3. Temperature Schedules")
print("-" * 40)
initial_temp = 1.0
final_temp = 0.0
total_steps = 10

print(f"Cooling from T={initial_temp} to T={final_temp} over {total_steps} steps")
print("\nLinear schedule:")
for step in range(total_steps + 1):
    temp = tlm.temperature_schedule(
        initial_temp, final_temp, step, total_steps, schedule_type='linear'
    )
    print(f"  Step {step:2d}: T = {temp:.3f}")

print("\nExponential schedule:")
initial_temp = 1.0
final_temp = 0.01  # Must be > 0 for exponential
for step in [0, 2, 4, 6, 8, 10]:
    temp = tlm.temperature_schedule(
        initial_temp, final_temp, step, total_steps, schedule_type='exponential'
    )
    print(f"  Step {step:2d}: T = {temp:.3f}")

print("\nCosine schedule:")
initial_temp = 1.0
final_temp = 0.0
for step in [0, 2, 4, 6, 8, 10]:
    temp = tlm.temperature_schedule(
        initial_temp, final_temp, step, total_steps, schedule_type='cosine'
    )
    print(f"  Step {step:2d}: T = {temp:.3f}")

# Example 4: Temperature Argmax
print("\n4. Temperature-Controlled Selection")
print("-" * 40)
logits = [2.0, 1.5, 1.0]

# Deterministic selection
idx_cold = tlm.temperature_argmax(logits, temperature=0.01)
print(f"Logits: {logits}")
print(f"T=0.01 selected index: {idx_cold} (value: {logits[idx_cold]})")

# Standard selection
idx_normal = tlm.temperature_argmax(logits, temperature=1.0)
print(f"T=1.0  selected index: {idx_normal} (value: {logits[idx_normal]})")

# Example 5: Practical Use Case - Document Ranking
print("\n5. Practical Example: Document Ranking with Temperature")
print("-" * 40)
doc_scores = [0.95, 0.92, 0.90, 0.50, 0.30]
doc_names = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]

print("Original document scores:")
for name, score in zip(doc_names, doc_scores):
    print(f"  {name}: {score:.2f}")

# Apply different temperatures
for temp in [0.1, 1.0, 3.0]:
    scaled = tlm.apply_temperature(doc_scores, temperature=temp)
    # Create ranking tuples
    ranking = sorted(zip(doc_names, scaled), key=lambda x: x[1], reverse=True)
    print(f"\nT={temp} ranking:")
    for name, score in ranking[:3]:  # Show top 3
        print(f"  {name}: {score:.3f}")

print("\n" + "=" * 60)
print("All temperature examples completed!")
print("=" * 60)
