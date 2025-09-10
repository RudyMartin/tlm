from __future__ import annotations
import math, random, builtins
from typing import List, Sequence, Tuple, Callable, Union

Scalar = Union[int, float]
Vector = List[Scalar]
Matrix = List[Vector]

# -----------------------
# Shapes & constructors
# -----------------------

def shape(x: Union[Scalar, Vector, Matrix]) -> Tuple[int, ...]:
    """Get shape of scalar, vector, or matrix."""
    if not isinstance(x, list):
        return ()
    if len(x) == 0:
        return (0,)
    if isinstance(x[0], list):
        return (len(x), len(x[0]))
    return (len(x),)

def array(obj: Union[Scalar, Sequence, Sequence[Sequence]]) -> Union[Scalar, Vector, Matrix]:
    """Convert sequence to array format."""
    if isinstance(obj, list):
        return [array(e) for e in obj]
    if isinstance(obj, tuple):
        return [array(e) for e in list(obj)]
    if isinstance(obj, (int, float)):
        return float(obj)
    return obj

def zeros(sz: Union[int, Tuple[int, int]]) -> Union[Vector, Matrix]:
    """Create array filled with zeros."""
    if isinstance(sz, int):
        return [0.0] * sz
    n, m = sz
    return [[0.0 for _ in range(m)] for _ in range(n)]

def ones(sz: Union[int, Tuple[int, int]]) -> Union[Vector, Matrix]:
    """Create array filled with ones."""
    if isinstance(sz, int):
        return [1.0] * sz
    n, m = sz
    return [[1.0 for _ in range(m)] for _ in range(n)]

def eye(n: int) -> Matrix:
    """Create identity matrix."""
    I = zeros((n, n))
    for i in range(n):
        I[i][i] = 1.0
    return I

# -----------------------
# Elementwise helpers
# -----------------------

def _is_scalar(x):
    return not isinstance(x, list)

def _is_vec(x):
    return isinstance(x, list) and (len(x) == 0 or not isinstance(x[0], list))

def _is_mat(x):
    return isinstance(x, list) and (len(x) > 0 and isinstance(x[0], list))

def _apply1(x, f: Callable[[Scalar], Scalar]):
    if isinstance(x, list):
        return [_apply1(e, f) for e in x]
    return f(float(x))

def _apply2(x, y, f: Callable[[Scalar, Scalar], Scalar]):
    if not isinstance(x, list) and not isinstance(y, list):
        return f(float(x), float(y))
    if not isinstance(x, list):
        return [_apply2(x, e, f) for e in y]
    if not isinstance(y, list):
        return [_apply2(e, y, f) for e in x]
    assert len(x) == len(y), "shape mismatch"
    return [_apply2(a, b, f) for a, b in zip(x, y)]

# public elementwise ops
def add(x, y):
    """Add two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a + b)

def sub(x, y):
    """Subtract two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a - b)

def mul(x, y):
    """Multiply two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a * b)

def div(x, y):
    """Divide two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a / b)

def exp(x):
    """Apply exponential function elementwise."""
    return _apply1(x, math.exp)

def log(x):
    """Apply natural logarithm elementwise."""
    return _apply1(x, lambda z: -1e30 if z <= 0 else math.log(z))

def sqrt(x):
    """Apply square root elementwise."""
    return _apply1(x, math.sqrt)

def clip(x, lo, hi):
    """Clip values to range [lo, hi]."""
    return _apply1(x, lambda z: hi if z > hi else (lo if z < lo else z))

# -----------------------
# Reductions
# -----------------------

def sum(x, axis: int | None = None) -> Union[Scalar, Vector]:
    """Compute sum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(_sum1d(x))
        elif _is_mat(x):
            return float(sum(_sum1d(row) for row in x))
        else:
            return float(x)
    else:
        if axis == 0 and _is_mat(x):
            n, m = shape(x)
            return [float(sum(x[i][j] for i in range(n))) for j in range(m)]
        if axis == 1 and _is_mat(x):
            return [float(_sum1d(row)) for row in x]
        raise ValueError("axis out of range or invalid for input")

def _sum1d(v: Vector) -> float:
    s = 0.0
    for a in v:
        s += float(a)
    return s

def mean(x, axis: int | None = None):
    """Compute mean along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return sum(x) / len(x)
        elif _is_mat(x):
            n, m = shape(x)
            return sum(x) / (n * m)
        else:
            return float(x)
    if axis == 0:
        n, m = shape(x)
        return [sum([x[i][j] for i in range(n)]) / n for j in range(m)]
    if axis == 1:
        return [sum(row) / len(row) for row in x]
    raise ValueError("axis out of range")

def var(x, axis: int | None = None, ddof: int = 0):
    """Compute variance along axis or all elements."""
    if axis is None:
        mu = mean(x)
        if _is_vec(x):
            return sum([(a - mu) ** 2 for a in x]) / (len(x) - ddof)
        elif _is_mat(x):
            n, m = shape(x)
            return sum([(a - mu) ** 2 for row in x for a in row]) / (n * m - ddof)
    else:
        if axis == 0:
            mu = mean(x, axis=0)
            n, m = shape(x)
            return [sum([(x[i][j] - mu[j]) ** 2 for i in range(n)]) / (n - ddof) for j in range(m)]
        if axis == 1:
            mus = mean(x, axis=1)
            return [sum([(a - mus[i]) ** 2 for a in row]) / (len(row) - ddof) for i, row in enumerate(x)]
    raise ValueError("axis out of range")

def std(x, axis: int | None = None, ddof: int = 0):
    """Compute standard deviation along axis or all elements."""
    if axis is None:
        return math.sqrt(var(x, axis=None, ddof=ddof))
    v = var(x, axis=axis, ddof=ddof)
    if isinstance(v, list):
        return [math.sqrt(a) for a in v]
    return math.sqrt(v)

def max(x, axis: int | None = None):
    """Compute maximum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(builtins.max(x))
        elif _is_mat(x):
            return float(builtins.max([builtins.max(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(builtins.max([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(builtins.max(row)) for row in x]
    raise ValueError("axis out of range")

def min(x, axis: int | None = None):
    """Compute minimum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(builtins.min(x))
        elif _is_mat(x):
            return float(builtins.min([builtins.min(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(builtins.min([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(builtins.min(row)) for row in x]
    raise ValueError("axis out of range")

def argmax(x, axis: int = -1):
    """Compute indices of maximum values along axis."""
    if _is_vec(x):
        return int(max(range(len(x)), key=lambda i: x[i]))
    if _is_mat(x):
        if axis in (-1, 1):
            return [int(max(range(len(row)), key=lambda j: row[j])) for row in x]
        if axis == 0:
            n, m = shape(x)
            return [int(max(range(n), key=lambda i: x[i][j])) for j in range(m)]
    raise ValueError("axis out of range or invalid for input")

# -----------------------
# Linear algebra basics
# -----------------------

def transpose(A: Matrix) -> Matrix:
    """Transpose a matrix."""
    n, m = shape(A)
    return [[A[i][j] for i in range(n)] for j in range(m)]

def dot(a: Vector, b: Vector) -> float:
    """Compute dot product of two vectors."""
    assert len(a) == len(b), "shape mismatch for dot"
    return sum([a[i] * b[i] for i in range(len(a))])

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Compute matrix multiplication."""
    n, k = shape(A)
    k2, m = shape(B)
    assert k == k2, "shape mismatch for matmul"
    BT = transpose(B)
    C = zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i][j] = dot(A[i], BT[j])
    return C

def norm(v: Vector, ord: int = 2) -> float:
    """Compute vector norm."""
    if ord == 2:
        return math.sqrt(sum([x*x for x in v]))
    if ord == 1:
        return sum([abs(x) for x in v])
    raise NotImplementedError

def l2_normalize(X, axis: int = -1, eps: float = 1e-12):
    """Apply L2 normalization along axis."""
    if _is_vec(X):
        nrm = builtins.max(norm(X), eps)
        return [x / nrm for x in X]
    out = []
    for row in X:
        nrm = builtins.max(norm(row), eps)
        out.append([x / nrm for x in row])
    return out

# -----------------------
# Shape manipulation
# -----------------------

def flatten(x: Union[Vector, Matrix]) -> Vector:
    """Flatten array to 1D."""
    if _is_vec(x):
        return x[:]
    result = []
    for row in x:
        result.extend(row)
    return result

def reshape(x: Union[Vector, Matrix], new_shape: Union[int, Tuple[int, ...]]) -> Union[Vector, Matrix]:
    """Reshape array to new dimensions."""
    # Flatten first
    flat = flatten(x) if _is_mat(x) else x[:]
    
    # Handle single dimension
    if isinstance(new_shape, int):
        if new_shape == -1:
            return flat
        if new_shape != len(flat):
            raise ValueError(f"Cannot reshape array of size {len(flat)} into shape ({new_shape},)")
        return flat
    
    # Handle 2D reshape
    if len(new_shape) == 1:
        if new_shape[0] == -1:
            return flat
        if new_shape[0] != len(flat):
            raise ValueError(f"Cannot reshape array of size {len(flat)} into shape {new_shape}")
        return flat
    
    if len(new_shape) == 2:
        n, m = new_shape
        total = len(flat)
        
        # Handle -1 (infer dimension)
        if n == -1:
            if total % m != 0:
                raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
            n = total // m
        elif m == -1:
            if total % n != 0:
                raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
            m = total // n
        
        if n * m != total:
            raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
        
        result = []
        for i in range(n):
            result.append(flat[i*m:(i+1)*m])
        return result
    
    raise NotImplementedError("Only 1D and 2D reshapes supported")

def concatenate(arrays: List[Union[Vector, Matrix]], axis: int = 0) -> Union[Vector, Matrix]:
    """Concatenate arrays along axis."""
    if not arrays:
        return []
    
    # All vectors
    if all(_is_vec(a) for a in arrays):
        if axis != 0:
            raise ValueError("Can only concatenate 1D arrays along axis 0")
        result = []
        for arr in arrays:
            result.extend(arr)
        return result
    
    # All matrices
    if all(_is_mat(a) for a in arrays):
        if axis == 0:
            # Stack rows
            result = []
            for arr in arrays:
                result.extend(arr)
            return result
        elif axis == 1:
            # Stack columns
            if not all(len(a) == len(arrays[0]) for a in arrays):
                raise ValueError("All arrays must have same number of rows for axis=1 concatenation")
            result = []
            for i in range(len(arrays[0])):
                row = []
                for arr in arrays:
                    row.extend(arr[i])
                result.append(row)
            return result
        else:
            raise ValueError(f"Invalid axis {axis} for 2D concatenation")
    
    raise ValueError("All arrays must have same dimensionality")

# -----------------------
# Distance & Similarity Metrics
# -----------------------

def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    
    dot_product = builtins.sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(builtins.sum(x * x for x in a))
    norm_b = math.sqrt(builtins.sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(a: Vector, b: Vector) -> float:
    """Compute Euclidean distance between two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    return math.sqrt(builtins.sum((x - y) ** 2 for x, y in zip(a, b)))

def manhattan_distance(a: Vector, b: Vector) -> float:
    """Compute Manhattan (L1) distance between two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    return builtins.sum(builtins.abs(x - y) for x, y in zip(a, b))

def hamming_distance(a: Vector, b: Vector) -> int:
    """Compute Hamming distance between two vectors."""
    assert len(a) == len(b), "Vectors must have same length"
    return builtins.sum(1 for x, y in zip(a, b) if x != y)

# -----------------------
# Additional Math Operations
# -----------------------

def power(x, p):
    """Raise elements to power p."""
    return _apply1(x, lambda z: z ** p)

def abs(x):
    """Compute absolute value elementwise."""
    return _apply1(x, lambda z: builtins.abs(z))

def sign(x):
    """Compute sign of elements (-1, 0, 1)."""
    return _apply1(x, lambda z: 1 if z > 0 else (-1 if z < 0 else 0))

# -----------------------
# Statistical Functions
# -----------------------

def median(x: Union[Vector, Matrix], axis: int | None = None) -> Union[float, Vector]:
    """Compute median along axis or all elements."""
    if axis is None:
        # Flatten all elements and find median
        if _is_vec(x):
            sorted_x = sorted(x)
        elif _is_mat(x):
            sorted_x = sorted([val for row in x for val in row])
        else:
            return float(x)
        
        n = len(sorted_x)
        if n % 2 == 0:
            return (sorted_x[n//2 - 1] + sorted_x[n//2]) / 2.0
        return float(sorted_x[n//2])
    
    if _is_mat(x):
        if axis == 0:
            # Median across rows for each column
            n, m = shape(x)
            result = []
            for j in range(m):
                col = sorted([x[i][j] for i in range(n)])
                if n % 2 == 0:
                    result.append((col[n//2 - 1] + col[n//2]) / 2.0)
                else:
                    result.append(float(col[n//2]))
            return result
        elif axis == 1:
            # Median across columns for each row
            result = []
            for row in x:
                sorted_row = sorted(row)
                m = len(sorted_row)
                if m % 2 == 0:
                    result.append((sorted_row[m//2 - 1] + sorted_row[m//2]) / 2.0)
                else:
                    result.append(float(sorted_row[m//2]))
            return result
    
    raise ValueError("Invalid axis for input")

def percentile(x: Union[Vector, Matrix], q: float, axis: int | None = None) -> Union[float, Vector]:
    """Compute the q-th percentile (0-100) along axis or all elements."""
    assert 0 <= q <= 100, "Percentile must be between 0 and 100"
    
    if axis is None:
        # Flatten all elements
        if _is_vec(x):
            sorted_x = sorted(x)
        elif _is_mat(x):
            sorted_x = sorted([val for row in x for val in row])
        else:
            return float(x)
        
        n = len(sorted_x)
        k = (n - 1) * q / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_x[int(k)]
        
        d0 = sorted_x[int(f)] * (c - k)
        d1 = sorted_x[int(c)] * (k - f)
        return d0 + d1
    
    if _is_mat(x):
        if axis == 0:
            # Percentile across rows for each column
            n, m = shape(x)
            result = []
            for j in range(m):
                col = sorted([x[i][j] for i in range(n)])
                k = (n - 1) * q / 100.0
                f = math.floor(k)
                c = math.ceil(k)
                
                if f == c:
                    result.append(col[int(k)])
                else:
                    d0 = col[int(f)] * (c - k)
                    d1 = col[int(c)] * (k - f)
                    result.append(d0 + d1)
            return result
        elif axis == 1:
            # Percentile across columns for each row
            result = []
            for row in x:
                sorted_row = sorted(row)
                m = len(sorted_row)
                k = (m - 1) * q / 100.0
                f = math.floor(k)
                c = math.ceil(k)
                
                if f == c:
                    result.append(sorted_row[int(k)])
                else:
                    d0 = sorted_row[int(f)] * (c - k)
                    d1 = sorted_row[int(c)] * (k - f)
                    result.append(d0 + d1)
            return result
    
    raise ValueError("Invalid axis for input")

def correlation(x: Vector, y: Vector) -> float:
    """Compute Pearson correlation coefficient between two vectors."""
    assert len(x) == len(y), "Vectors must have same length"
    
    n = len(x)
    if n == 0:
        return 0.0
    
    mean_x = builtins.sum(x) / n
    mean_y = builtins.sum(y) / n
    
    cov = builtins.sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(builtins.sum((x[i] - mean_x) ** 2 for i in range(n)))
    std_y = math.sqrt(builtins.sum((y[i] - mean_y) ** 2 for i in range(n)))
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)

def covariance_matrix(X: Matrix) -> Matrix:
    """Compute covariance matrix of data matrix X (samples x features)."""
    n, m = shape(X)
    
    # Center the data
    means = mean(X, axis=0)
    X_centered = [[X[i][j] - means[j] for j in range(m)] for i in range(n)]
    
    # Compute covariance matrix
    cov = zeros((m, m))
    for i in range(m):
        for j in range(m):
            cov[i][j] = builtins.sum(X_centered[k][i] * X_centered[k][j] for k in range(n)) / (n - 1)
    
    return cov

# -----------------------
# Array Manipulation
# -----------------------

def unique(x: Union[Vector, Matrix]) -> Vector:
    """Get unique elements from array."""
    if _is_vec(x):
        seen = set()
        result = []
        for val in x:
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result
    elif _is_mat(x):
        seen = set()
        result = []
        for row in x:
            for val in row:
                if val not in seen:
                    seen.add(val)
                    result.append(val)
        return result
    return [x]

def where(condition: Union[Vector, Matrix], x=None, y=None) -> Union[List[int], Union[Vector, Matrix]]:
    """Return indices where condition is True, or choose x/y based on condition."""
    if x is None and y is None:
        # Return indices where condition is True
        if _is_vec(condition):
            return [i for i, val in enumerate(condition) if val]
        elif _is_mat(condition):
            indices = []
            for i, row in enumerate(condition):
                for j, val in enumerate(row):
                    if val:
                        indices.append([i, j])
            return indices
        return []
    
    # Choose between x and y based on condition
    if _is_vec(condition):
        return [x[i] if condition[i] else y[i] for i in range(len(condition))]
    elif _is_mat(condition):
        result = []
        for i, row in enumerate(condition):
            result.append([x[i][j] if condition[i][j] else y[i][j] for j in range(len(row))])
        return result
    
    return x if condition else y

def tile(x: Union[Vector, Matrix], reps: Union[int, Tuple[int, int]]) -> Union[Vector, Matrix]:
    """Repeat array along axes."""
    if isinstance(reps, int):
        # 1D tiling
        if _is_vec(x):
            result = []
            for _ in range(reps):
                result.extend(x)
            return result
        elif _is_mat(x):
            result = []
            for _ in range(reps):
                result.extend(x)
            return result
    elif isinstance(reps, tuple) and len(reps) == 2:
        # 2D tiling
        reps_y, reps_x = reps
        if _is_vec(x):
            # Treat vector as row vector for 2D tiling
            result = []
            row = []
            for _ in range(reps_x):
                row.extend(x)
            for _ in range(reps_y):
                result.append(row[:])
            return result
        elif _is_mat(x):
            result = []
            for _ in range(reps_y):
                for row in x:
                    new_row = []
                    for _ in range(reps_x):
                        new_row.extend(row)
                    result.append(new_row)
            return result
    
    raise ValueError("Invalid reps argument")

def stack(arrays: List[Vector], axis: int = 0) -> Matrix:
    """Stack vectors into matrix along new axis."""
    if not arrays:
        return []
    
    if axis == 0:
        # Stack as rows
        return [arr[:] for arr in arrays]
    elif axis == 1:
        # Stack as columns
        n = len(arrays[0])
        if not all(len(arr) == n for arr in arrays):
            raise ValueError("All arrays must have same length for axis=1 stacking")
        return transpose([arr[:] for arr in arrays])
    else:
        raise ValueError(f"Invalid axis {axis}")

def vstack(arrays: List[Union[Vector, Matrix]]) -> Matrix:
    """Stack arrays vertically (row-wise)."""
    result = []
    for arr in arrays:
        if _is_vec(arr):
            result.append(arr[:])
        elif _is_mat(arr):
            result.extend(arr)
        else:
            result.append([arr])
    return result

def hstack(arrays: List[Union[Vector, Matrix]]) -> Union[Vector, Matrix]:
    """Stack arrays horizontally (column-wise)."""
    if all(_is_vec(arr) for arr in arrays):
        # Concatenate vectors
        result = []
        for arr in arrays:
            result.extend(arr)
        return result
    
    # Convert all to matrices and stack columns
    matrices = []
    for arr in arrays:
        if _is_vec(arr):
            # Convert vector to column matrix
            matrices.append([[val] for val in arr])
        elif _is_mat(arr):
            matrices.append(arr)
        else:
            matrices.append([[arr]])
    
    # Check all have same number of rows
    n_rows = len(matrices[0])
    if not all(len(m) == n_rows for m in matrices):
        raise ValueError("All arrays must have same number of rows for horizontal stacking")
    
    # Concatenate columns
    result = []
    for i in range(n_rows):
        row = []
        for m in matrices:
            row.extend(m[i])
        result.append(row)
    
    return result

# -----------------------
# Special Functions for Transformers
# -----------------------

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> Matrix:
    """Generate sinusoidal positional encoding for transformer models."""
    encoding = zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                # Even indices: sin
                encoding[pos][i] = math.sin(pos / (base ** (i / d_model)))
            else:
                # Odd indices: cos
                encoding[pos][i] = math.cos(pos / (base ** ((i - 1) / d_model)))
    
    return encoding


# -----------------------
# Missing Core Functions 
# -----------------------

def histogram(x: Vector, bins: Union[int, Vector] = 10, range_: Optional[Tuple[float, float]] = None) -> Tuple[Vector, Vector]:
    """
    Compute histogram of data.
    
    Args:
        x: Input data
        bins: Number of bins (int) or bin edges (list)
        range_: (min, max) range for bins
        
    Returns:
        (counts, bin_edges)
    """
    if not x:
        return [], []
        
    if isinstance(bins, int):
        # Auto-generate bin edges
        if range_ is None:
            x_min, x_max = min(x), max(x)
        else:
            x_min, x_max = range_
            
        if x_max == x_min:
            x_max = x_min + 1.0  # Handle constant data
            
        width = (x_max - x_min) / bins
        bin_edges = [x_min + i * width for i in range(bins + 1)]
    else:
        # User-provided bin edges
        bin_edges = list(bins)
        bins = len(bin_edges) - 1
    
    # Count values in each bin
    counts = [0] * bins
    for val in x:
        # Find which bin this value belongs to
        for i in range(bins):
            if bin_edges[i] <= val < bin_edges[i + 1]:
                counts[i] += 1
                break
            elif i == bins - 1 and val == bin_edges[i + 1]:
                # Handle edge case: value equals maximum
                counts[i] += 1
                break
    
    return counts, bin_edges


def allclose(x: Union[Vector, Matrix], y: Union[Vector, Matrix], 
             rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Test if arrays are element-wise close within tolerances.
    
    Returns True if: |x - y| <= atol + rtol * |y|
    """
    if shape(x) != shape(y):
        return False
        
    def _close_scalar(a: float, b: float) -> bool:
        return abs(a - b) <= atol + rtol * abs(b)
    
    if _is_scalar(x):
        return _close_scalar(x, y)
    elif _is_vec(x):
        return all(_close_scalar(a, b) for a, b in zip(x, y))
    elif _is_mat(x):
        for i in range(len(x)):
            if not all(_close_scalar(a, b) for a, b in zip(x[i], y[i])):
                return False
        return True
    
    return False


def searchsorted(a: Vector, v: Union[float, Vector], side: str = 'left') -> Union[int, Vector]:
    """
    Find indices to insert v in sorted array a to maintain order.
    
    Args:
        a: Sorted array
        v: Values to insert  
        side: 'left' or 'right' for tie-breaking
        
    Returns:
        Insertion indices
    """
    def _searchsorted_scalar(arr: Vector, val: float, left: bool = True) -> int:
        """Binary search for insertion point."""
        low, high = 0, len(arr)
        
        while low < high:
            mid = (low + high) // 2
            if left:
                if arr[mid] < val:
                    low = mid + 1
                else:
                    high = mid
            else:  # right side
                if arr[mid] <= val:
                    low = mid + 1
                else:
                    high = mid
        return low
    
    left_side = (side == 'left')
    
    if _is_scalar(v):
        return _searchsorted_scalar(a, v, left_side)
    else:
        return [_searchsorted_scalar(a, val, left_side) for val in v]


def diff(x: Union[Vector, Matrix], n: int = 1, axis: Optional[int] = None) -> Union[Vector, Matrix]:
    """
    Calculate n-th discrete difference along given axis.
    
    Args:
        x: Input array
        n: Number of times to difference (default 1)
        axis: Axis along which to difference (None for 1D)
        
    Returns:
        Differenced array
    """
    if n < 0:
        raise ValueError("diff order must be non-negative")
    if n == 0:
        return x
        
    if _is_vec(x):
        result = x
        for _ in range(n):
            result = [result[i+1] - result[i] for i in range(len(result)-1)]
        return result
    elif _is_mat(x):
        if axis == 0 or axis is None:
            # Difference along rows (subtract row i from row i+1)
            result = x
            for _ in range(n):
                new_result = []
                for i in range(len(result)-1):
                    diff_row = [result[i+1][j] - result[i][j] for j in range(len(result[i]))]
                    new_result.append(diff_row)
                result = new_result
            return result
        elif axis == 1:
            # Difference along columns
            result = x
            for _ in range(n):
                new_result = []
                for row in result:
                    diff_row = [row[j+1] - row[j] for j in range(len(row)-1)]
                    new_result.append(diff_row)
                result = new_result
            return result
        else:
            raise ValueError("axis must be 0 or 1 for 2D arrays")
    
    return x


def gradient(x: Union[Vector, Matrix], *varargs, axis: Optional[int] = None, 
             edge_order: int = 1) -> Union[Vector, Matrix, List]:
    """
    Calculate numerical gradient using central differences.
    
    Args:
        x: Input array
        *varargs: Spacing between points (default 1)
        axis: Axis for gradient (None means all axes for 2D)
        edge_order: Accuracy order at boundaries (1 or 2)
        
    Returns:
        Gradient array(s)
    """
    if edge_order not in [1, 2]:
        raise ValueError("edge_order must be 1 or 2")
        
    def _gradient_1d(arr: Vector, spacing: float = 1.0) -> Vector:
        """Calculate 1D gradient."""
        n = len(arr)
        if n < 2:
            return [0.0] * n
            
        grad = [0.0] * n
        
        # Forward difference at start
        if edge_order == 1:
            grad[0] = (arr[1] - arr[0]) / spacing
        else:  # edge_order == 2
            if n >= 3:
                grad[0] = (-3*arr[0] + 4*arr[1] - arr[2]) / (2 * spacing)
            else:
                grad[0] = (arr[1] - arr[0]) / spacing
        
        # Central differences in middle
        for i in range(1, n-1):
            grad[i] = (arr[i+1] - arr[i-1]) / (2 * spacing)
        
        # Backward difference at end
        if edge_order == 1:
            grad[n-1] = (arr[n-1] - arr[n-2]) / spacing  
        else:  # edge_order == 2
            if n >= 3:
                grad[n-1] = (arr[n-3] - 4*arr[n-2] + 3*arr[n-1]) / (2 * spacing)
            else:
                grad[n-1] = (arr[n-1] - arr[n-2]) / spacing
                
        return grad
    
    # Handle spacing arguments
    spacing = varargs[0] if varargs else 1.0
    
    if _is_vec(x):
        return _gradient_1d(x, spacing)
    elif _is_mat(x):
        if axis == 0:
            # Gradient along rows
            result = []
            for j in range(len(x[0])):  # For each column
                column = [x[i][j] for i in range(len(x))]
                grad_col = _gradient_1d(column, spacing)
                result.append(grad_col)
            # Transpose back
            return [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
        elif axis == 1:
            # Gradient along columns  
            return [_gradient_1d(row, spacing) for row in x]
        else:
            # Return gradients for both axes
            grad_0 = gradient(x, spacing, axis=0, edge_order=edge_order)
            grad_1 = gradient(x, spacing, axis=1, edge_order=edge_order) 
            return [grad_0, grad_1]
    
    return x


def convolve(x: Vector, y: Vector, mode: str = 'full') -> Vector:
    """
    Discrete convolution of two sequences.
    
    Args:
        x, y: Input sequences  
        mode: 'full', 'same', or 'valid'
        
    Returns:
        Convolved sequence
    """
    if not x or not y:
        return []
        
    n, m = len(x), len(y)
    
    if mode == 'full':
        # Full convolution: length n + m - 1
        result = [0.0] * (n + m - 1)
        for i in range(n):
            for j in range(m):
                result[i + j] += x[i] * y[j]
        return result
        
    elif mode == 'same':
        # Same size as larger input
        full_conv = convolve(x, y, 'full')
        if n >= m:
            # Same size as x
            start = (len(full_conv) - n) // 2
            return full_conv[start:start + n]
        else:
            # Same size as y  
            start = (len(full_conv) - m) // 2
            return full_conv[start:start + m]
            
    elif mode == 'valid':
        # Only where sequences fully overlap
        if n < m:
            x, y, n, m = y, x, m, n  # Ensure x is longer
        if m == 0:
            return []
        result = [0.0] * (n - m + 1)
        for i in range(n - m + 1):
            for j in range(m):
                result[i] += x[i + j] * y[j]
        return result
        
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")


# Enhanced numerical validation functions  
def isfinite(x: Union[float, Vector, Matrix]) -> Union[bool, Vector, Matrix]:
    """Test element-wise for finite values."""
    import math
    
    def _isfinite_scalar(val):
        return not (math.isinf(val) or math.isnan(val))
    
    if _is_scalar(x):
        return _isfinite_scalar(x)
    elif _is_vec(x):
        return [_isfinite_scalar(val) for val in x]
    elif _is_mat(x):
        return [[_isfinite_scalar(val) for val in row] for row in x]
    
    return True


def isinf(x: Union[float, Vector, Matrix]) -> Union[bool, Vector, Matrix]:
    """Test element-wise for infinite values."""
    import math
    
    if _is_scalar(x):
        return math.isinf(x)
    elif _is_vec(x):
        return [math.isinf(val) for val in x]
    elif _is_mat(x):
        return [[math.isinf(val) for val in row] for row in x]
    
    return False


def isnan(x: Union[float, Vector, Matrix]) -> Union[bool, Vector, Matrix]:
    """Test element-wise for NaN values."""  
    import math
    
    if _is_scalar(x):
        return math.isnan(x)
    elif _is_vec(x):
        return [math.isnan(val) for val in x]
    elif _is_mat(x):
        return [[math.isnan(val) for val in row] for row in x]
    
    return False
