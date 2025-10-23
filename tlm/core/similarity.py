"""
Similarity and distance metrics.

Pure Python implementations for vector similarity calculations.
Built on top of tlm.pure.ops for core similarity functions.
"""

from ..pure.ops import cosine_similarity as _cosine_similarity
from ..pure.ops import euclidean_distance as _euclidean_distance
from ..pure.ops import manhattan_distance as _manhattan_distance
from math import sqrt


# Re-export basic functions from pure.ops
cosine_similarity = _cosine_similarity
euclidean_distance = _euclidean_distance
manhattan_distance = _manhattan_distance


def pairwise_cosine(vectors):
    """Compute pairwise cosine similarity matrix.

    Args:
        vectors: List of vectors (list of lists)

    Returns:
        2D list where result[i][j] is similarity between vectors[i] and vectors[j]

    Examples:
        >>> vecs = [[1, 0], [0, 1], [1, 1]]
        >>> matrix = pairwise_cosine(vecs)
        >>> matrix[0][1]  # Similarity between [1,0] and [0,1]
        0.0
    """
    # Get number of vectors to create n x n similarity matrix
    n = len(vectors)

    # Initialize result matrix with zeros
    result = [[0.0] * n for _ in range(n)]

    # Compute similarities - only upper triangle since matrix is symmetric
    for i in range(n):
        for j in range(i, n):
            # Calculate cosine similarity between vector i and vector j
            sim = cosine_similarity(vectors[i], vectors[j])

            # Fill both [i][j] and [j][i] since cosine similarity is symmetric
            result[i][j] = sim
            result[j][i] = sim  # Mirror across diagonal

    return result


def top_k_similar(query_vec, corpus_vecs, k=5):
    """Find k most similar vectors from corpus.

    Args:
        query_vec: Query vector
        corpus_vecs: List of corpus vectors
        k: Number of results to return

    Returns:
        List of (index, similarity_score) tuples, sorted by score descending

    Examples:
        >>> query = [1, 0, 0]
        >>> corpus = [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]
        >>> results = top_k_similar(query, corpus, k=2)
        >>> results[0][0]  # Index of most similar
        0
    """
    # Compute cosine similarity between query and each corpus vector
    similarities = []
    for idx, vec in enumerate(corpus_vecs):
        # Calculate similarity for this corpus vector
        sim = cosine_similarity(query_vec, vec)
        # Store (index, score) tuple for later sorting
        similarities.append((idx, sim))

    # Sort by similarity score in descending order (highest similarity first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return only the top k most similar results
    return similarities[:k]


def pairwise_distances(vectors, metric='euclidean'):
    """Compute pairwise distance matrix using specified metric.

    Args:
        vectors: List of vectors
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')

    Returns:
        2D list where result[i][j] is distance between vectors[i] and vectors[j]

    Examples:
        >>> vecs = [[0, 0], [3, 4], [1, 1]]
        >>> matrix = pairwise_distances(vecs, metric='euclidean')
        >>> matrix[0][1]  # Distance between [0,0] and [3,4]
        5.0
    """
    # Get number of vectors for n x n distance matrix
    n = len(vectors)

    # Initialize result matrix with zeros (diagonal will remain 0)
    result = [[0.0] * n for _ in range(n)]

    # Select distance function based on metric parameter
    if metric == 'euclidean':
        dist_func = euclidean_distance  # L2 norm
    elif metric == 'manhattan':
        dist_func = manhattan_distance  # L1 norm
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine_similarity (converts similarity to distance)
        dist_func = lambda a, b: 1.0 - cosine_similarity(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute distances - only upper triangle since matrix is symmetric
    for i in range(n):
        for j in range(i + 1, n):  # Start at i+1 to skip diagonal
            # Calculate distance between vector i and vector j
            dist = dist_func(vectors[i], vectors[j])

            # Fill both [i][j] and [j][i] since distance is symmetric
            result[i][j] = dist
            result[j][i] = dist  # Mirror across diagonal

    return result


def nearest_neighbors(query_vec, corpus_vecs, k=5, metric='euclidean'):
    """Find k nearest neighbors using specified distance metric.

    Args:
        query_vec: Query vector
        corpus_vecs: List of corpus vectors
        k: Number of neighbors to return
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')

    Returns:
        List of (index, distance) tuples, sorted by distance ascending

    Examples:
        >>> query = [0, 0]
        >>> corpus = [[1, 0], [0, 1], [5, 5]]
        >>> results = nearest_neighbors(query, corpus, k=2, metric='euclidean')
        >>> results[0][0] in [0, 1]  # One of the close points
        True
    """
    # Select distance function based on metric parameter
    if metric == 'euclidean':
        dist_func = euclidean_distance  # L2 norm (straight-line distance)
    elif metric == 'manhattan':
        dist_func = manhattan_distance  # L1 norm (city-block distance)
    elif metric == 'cosine':
        # Convert cosine similarity to distance (smaller = more similar)
        dist_func = lambda a, b: 1.0 - cosine_similarity(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute distance from query to each corpus vector
    distances = []
    for idx, vec in enumerate(corpus_vecs):
        # Calculate distance for this corpus vector
        dist = dist_func(query_vec, vec)
        # Store (index, distance) tuple for later sorting
        distances.append((idx, dist))

    # Sort by distance in ascending order (smallest distance = nearest neighbor)
    distances.sort(key=lambda x: x[1])

    # Return only the k nearest neighbors
    return distances[:k]
