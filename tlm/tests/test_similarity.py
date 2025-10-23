"""
Tests for similarity metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tlm


def test_cosine_similarity_identical():
    """Test cosine similarity of identical vectors."""
    vec = [1.0, 2.0, 3.0]
    sim = tlm.cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6, f"Expected 1.0, got {sim}"
    print("✓ Cosine similarity of identical vectors = 1.0")


def test_cosine_similarity_orthogonal():
    """Test cosine similarity of orthogonal vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    sim = tlm.cosine_similarity(vec1, vec2)
    assert abs(sim - 0.0) < 1e-6, f"Expected 0.0, got {sim}"
    print("✓ Cosine similarity of orthogonal vectors = 0.0")


def test_cosine_similarity_opposite():
    """Test cosine similarity of opposite vectors."""
    vec1 = [1.0, 0.0]
    vec2 = [-1.0, 0.0]
    sim = tlm.cosine_similarity(vec1, vec2)
    assert abs(sim - (-1.0)) < 1e-6, f"Expected -1.0, got {sim}"
    print("✓ Cosine similarity of opposite vectors = -1.0")


def test_pairwise_cosine():
    """Test pairwise cosine similarity matrix."""
    vecs = [[1, 0], [0, 1], [1, 1]]
    matrix = tlm.pairwise_cosine(vecs)

    # Check diagonal (self-similarity = 1)
    assert abs(matrix[0][0] - 1.0) < 1e-6
    assert abs(matrix[1][1] - 1.0) < 1e-6
    print("✓ Pairwise cosine diagonal = 1.0")

    # Check orthogonal vectors
    assert abs(matrix[0][1] - 0.0) < 1e-6
    print("✓ Pairwise cosine for orthogonal vectors = 0.0")

    # Check symmetry
    assert abs(matrix[0][1] - matrix[1][0]) < 1e-6
    print("✓ Pairwise cosine matrix is symmetric")


def test_top_k_similar():
    """Test top-k similar vector retrieval."""
    query = [1.0, 0.0, 0.0]
    corpus = [
        [1.0, 0.0, 0.0],  # Identical (sim=1.0)
        [0.0, 1.0, 0.0],  # Orthogonal (sim=0.0)
        [0.9, 0.1, 0.0],  # Similar (sim≈0.99)
    ]

    results = tlm.top_k_similar(query, corpus, k=2)

    # Check we got 2 results
    assert len(results) == 2
    print("✓ Top-k returns correct number of results")

    # Check first result is index 0 (identical)
    assert results[0][0] == 0
    assert abs(results[0][1] - 1.0) < 1e-6
    print("✓ Top-1 result is most similar vector")

    # Check second result is index 2 (similar)
    assert results[1][0] == 2
    print("✓ Top-2 result is second most similar vector")


def test_euclidean_distance():
    """Test Euclidean distance calculation."""
    vec1 = [0, 0]
    vec2 = [3, 4]
    dist = tlm.euclidean_distance(vec1, vec2)
    assert abs(dist - 5.0) < 1e-6, f"Expected 5.0, got {dist}"
    print("✓ Euclidean distance [0,0] to [3,4] = 5.0")


def test_manhattan_distance():
    """Test Manhattan distance calculation."""
    vec1 = [0, 0]
    vec2 = [3, 4]
    dist = tlm.manhattan_distance(vec1, vec2)
    assert dist == 7.0, f"Expected 7.0, got {dist}"
    print("✓ Manhattan distance [0,0] to [3,4] = 7.0")


def test_pairwise_distances():
    """Test pairwise distance matrix."""
    vecs = [[0, 0], [3, 4], [1, 1]]
    matrix = tlm.pairwise_distances(vecs, metric='euclidean')

    # Check diagonal (self-distance = 0)
    assert abs(matrix[0][0]) < 1e-6
    print("✓ Pairwise distance diagonal = 0.0")

    # Check distance between [0,0] and [3,4]
    assert abs(matrix[0][1] - 5.0) < 1e-6
    print("✓ Pairwise distance [0,0] to [3,4] = 5.0")

    # Check symmetry
    assert abs(matrix[0][1] - matrix[1][0]) < 1e-6
    print("✓ Pairwise distance matrix is symmetric")


def test_nearest_neighbors():
    """Test nearest neighbors search."""
    query = [0, 0]
    corpus = [[1, 0], [0, 1], [5, 5]]

    results = tlm.nearest_neighbors(query, corpus, k=2, metric='euclidean')

    # Check we got 2 results
    assert len(results) == 2
    print("✓ Nearest neighbors returns correct number")

    # Check first two are closest (distance = 1.0)
    assert results[0][0] in [0, 1]
    assert abs(results[0][1] - 1.0) < 1e-6
    print("✓ Nearest neighbor has distance 1.0")

    # Check sorted by distance
    assert results[0][1] <= results[1][1]
    print("✓ Results sorted by distance ascending")


if __name__ == '__main__':
    print("\n=== Testing Similarity Functions ===\n")
    test_cosine_similarity_identical()
    test_cosine_similarity_orthogonal()
    test_cosine_similarity_opposite()
    test_pairwise_cosine()
    test_top_k_similar()
    test_euclidean_distance()
    test_manhattan_distance()
    test_pairwise_distances()
    test_nearest_neighbors()
    print("\n✅ All similarity tests passed!\n")
