"""
Example 1: Basic Similarity Functions

Demonstrates core similarity operations from tlm.core.similarity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tlm

print("=" * 60)
print("TLM Similarity Functions - Basic Examples")
print("=" * 60)

# Example 1: Cosine Similarity
print("\n1. Cosine Similarity")
print("-" * 40)
vec1 = [1, 0, 0]
vec2 = [0, 1, 0]
vec3 = [1, 1, 0]

sim_12 = tlm.cosine_similarity(vec1, vec2)
sim_13 = tlm.cosine_similarity(vec1, vec3)
sim_23 = tlm.cosine_similarity(vec2, vec3)

print(f"vec1: {vec1}")
print(f"vec2: {vec2}")
print(f"vec3: {vec3}")
print(f"\nSimilarity(vec1, vec2): {sim_12:.3f}")
print(f"Similarity(vec1, vec3): {sim_13:.3f}")
print(f"Similarity(vec2, vec3): {sim_23:.3f}")

# Example 2: Pairwise Similarity Matrix
print("\n2. Pairwise Similarity Matrix")
print("-" * 40)
documents = [
    [1.0, 0.0, 0.0],  # Document about topic A
    [0.0, 1.0, 0.0],  # Document about topic B
    [0.5, 0.5, 0.0],  # Document about both A and B
    [0.0, 0.0, 1.0],  # Document about topic C
]

similarity_matrix = tlm.pairwise_cosine(documents)

print("Document similarity matrix:")
for i, row in enumerate(similarity_matrix):
    print(f"Doc {i}: [{', '.join(f'{val:5.2f}' for val in row)}]")

# Example 3: Top-K Similar Documents
print("\n3. Top-K Similar Documents")
print("-" * 40)
query = [0.6, 0.4, 0.0]  # Query with both A and B
corpus = documents

results = tlm.top_k_similar(query, corpus, k=3)

print(f"Query vector: {query}")
print("\nTop 3 most similar documents:")
for rank, (idx, score) in enumerate(results, 1):
    print(f"  {rank}. Document {idx}: similarity = {score:.3f}")

# Example 4: Distance Metrics
print("\n4. Distance Metrics (Euclidean)")
print("-" * 40)
point1 = [0, 0]
point2 = [3, 4]
point3 = [1, 1]

dist_12 = tlm.euclidean_distance(point1, point2)
dist_13 = tlm.euclidean_distance(point1, point3)
dist_23 = tlm.euclidean_distance(point2, point3)

print(f"point1: {point1}")
print(f"point2: {point2}")
print(f"point3: {point3}")
print(f"\nDistance(point1, point2): {dist_12:.3f}")
print(f"Distance(point1, point3): {dist_13:.3f}")
print(f"Distance(point2, point3): {dist_23:.3f}")

# Example 5: Nearest Neighbors
print("\n5. Nearest Neighbors")
print("-" * 40)
query_point = [0, 0]
corpus_points = [
    [1, 0],
    [0, 1],
    [5, 5],
    [2, 2],
    [10, 10]
]

neighbors = tlm.nearest_neighbors(query_point, corpus_points, k=3, metric='euclidean')

print(f"Query point: {query_point}")
print(f"Corpus points: {corpus_points}")
print("\nNearest 3 neighbors:")
for rank, (idx, dist) in enumerate(neighbors, 1):
    print(f"  {rank}. Point {idx} ({corpus_points[idx]}): distance = {dist:.3f}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
