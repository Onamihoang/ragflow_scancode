# Vector Similarity - Phân Tích Toán Học Chi Tiết

Phân tích chi tiết các thuật toán tính độ tương tự giữa vectors trong RAG system.

**Files:**
- `rag/llm/embedding_model.py`
- `rag/utils/es_conn.py`

---

## 📐 Các Metrics Tương Tự

### **1. Cosine Similarity**

#### **Định Nghĩa**

$$
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}
$$

**Geometric Interpretation:**

```
     a
     ↗
    /  ) θ
   /___→ b

cos(θ) = adjacent/hypotenuse
```

#### **Properties**

- **Range:** [-1, 1]
  - cos(0°) = 1 → Same direction (identical)
  - cos(90°) = 0 → Orthogonal (unrelated)
  - cos(180°) = -1 → Opposite direction

- **Magnitude Invariant:**
  - cos(2a, 2b) = cos(a, b)
  - Only cares about direction, not length

#### **Implementation**

```python
import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors

    Args:
        a, b: numpy arrays of same shape

    Returns:
        float in [-1, 1]

    Complexity: O(n) where n = vector dimension
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)
```

#### **Example Calculation**

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Step 1: Dot product
dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

# Step 2: Norms
||a|| = sqrt(1² + 2² + 3²) = sqrt(14) ≈ 3.742
||b|| = sqrt(4² + 5² + 6²) = sqrt(77) ≈ 8.775

# Step 3: Cosine
cos(a, b) = 32 / (3.742 * 8.775) ≈ 0.9746

# Angle
θ = arccos(0.9746) ≈ 12.9°
```

---

### **2. Euclidean Distance**

#### **Định Nghĩa**

$$
d(\mathbf{a}, \mathbf{b}) = ||\mathbf{a} - \mathbf{b}|| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
$$

**Geometric Interpretation:**
```
  a •
     \
      \  d(a,b)
       \
        • b
```

#### **Properties**

- **Range:** [0, ∞)
  - d = 0 → Identical vectors
  - Larger d → More different

- **Magnitude Sensitive:**
  - d(2a, 2b) = 2 × d(a, b)
  - Affected by vector length

#### **Implementation**

```python
def euclidean_distance(a, b):
    """
    Compute Euclidean (L2) distance

    Complexity: O(n)
    """
    diff = a - b
    return np.sqrt(np.sum(diff ** 2))

# Optimized version
def euclidean_distance_fast(a, b):
    """Use numpy's built-in norm"""
    return np.linalg.norm(a - b)
```

#### **Example**

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

diff = [4-1, 5-2, 6-3] = [3, 3, 3]
d = sqrt(3² + 3² + 3²) = sqrt(27) ≈ 5.196
```

---

### **3. Dot Product**

#### **Định Nghĩa**

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = ||\mathbf{a}|| \cdot ||\mathbf{b}|| \cdot \cos(\theta)
$$

**For normalized vectors** (||a|| = ||b|| = 1):

$$
\mathbf{a} \cdot \mathbf{b} = \cos(\theta)
$$

#### **Properties**

- **Range:** (-∞, ∞)
  - For normalized: [-1, 1]

- **Fastest to compute:**
  - No square root needed
  - Just multiply & sum

#### **Implementation**

```python
def dot_product(a, b):
    """
    Compute dot product

    Complexity: O(n)
    """
    return np.dot(a, b)

# Or elementwise
def dot_product_manual(a, b):
    return np.sum(a * b)
```

---

### **4. Manhattan Distance (L1)**

#### **Định Nghĩa**

$$
d_{L1}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i|
$$

**Geometric Interpretation:**
```
  a •___
      |  \
      |   • b
```
Distance along grid lines (taxi-cab metric).

#### **Implementation**

```python
def manhattan_distance(a, b):
    """
    Compute L1 distance

    Complexity: O(n)
    """
    return np.sum(np.abs(a - b))
```

#### **Example**

```python
a = [1, 2, 3]
b = [4, 5, 6]

d_L1 = |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
```

---

## 🔬 Normalization Techniques

### **L2 Normalization**

#### **Formula**

$$
\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}
$$

where $||\mathbf{v}||_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$

#### **Implementation**

```python
def l2_normalize(v):
    """
    L2 normalize vector

    Args:
        v: numpy array

    Returns:
        Normalized vector with ||v|| = 1
    """
    norm = np.linalg.norm(v)

    if norm == 0:
        return v  # Avoid division by zero

    return v / norm
```

#### **Properties**

After normalization:
- $||\mathbf{v}_{\text{norm}}||_2 = 1$
- Cosine similarity = Dot product
- All vectors on unit hypersphere

#### **Verification**

```python
v = np.array([3, 4])
v_norm = l2_normalize(v)

print(v_norm)           # [0.6, 0.8]
print(np.linalg.norm(v_norm))  # 1.0 ✓

# Verify: 0.6² + 0.8² = 0.36 + 0.64 = 1.0 ✓
```

---

### **Min-Max Normalization**

#### **Formula**

$$
v_i' = \frac{v_i - \min(\mathbf{v})}{\max(\mathbf{v}) - \min(\mathbf{v})}
$$

**Range:** [0, 1]

#### **Implementation**

```python
def min_max_normalize(v):
    """
    Scale to [0, 1] range
    """
    v_min = np.min(v)
    v_max = np.max(v)

    if v_max == v_min:
        return np.zeros_like(v)

    return (v - v_min) / (v_max - v_min)
```

---

### **Z-Score Normalization**

#### **Formula**

$$
v_i' = \frac{v_i - \mu}{\sigma}
$$

where:
- $\mu = \frac{1}{n}\sum_{i=1}^{n} v_i$ (mean)
- $\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (v_i - \mu)^2}$ (std dev)

#### **Implementation**

```python
def z_score_normalize(v):
    """
    Standardize to mean=0, std=1
    """
    mean = np.mean(v)
    std = np.std(v)

    if std == 0:
        return v - mean

    return (v - mean) / std
```

---

## ⚡ Performance Optimization

### **Vectorized Operations**

#### **Bad: Loop**

```python
def cosine_similarity_slow(a, b):
    """O(n) but slow due to Python loop"""
    dot = 0
    norm_a = 0
    norm_b = 0

    for i in range(len(a)):
        dot += a[i] * b[i]
        norm_a += a[i] ** 2
        norm_b += b[i] ** 2

    norm_a = math.sqrt(norm_a)
    norm_b = math.sqrt(norm_b)

    return dot / (norm_a * norm_b)
```

**Benchmark:** ~100 µs for 768-dim vectors

#### **Good: Numpy**

```python
def cosine_similarity_fast(a, b):
    """O(n) but fast with SIMD"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**Benchmark:** ~5 µs for 768-dim vectors (20x faster!)

---

### **Batch Processing**

#### **Single**

```python
# Compute similarity for each pair
similarities = []
for q in queries:
    for d in docs:
        sim = cosine_similarity(q, d)
        similarities.append(sim)

# Time: N × M × O(d)
```

#### **Batch**

```python
# Use matrix multiplication
Q = np.array(queries)    # (N, d)
D = np.array(docs)       # (M, d)

# All pairwise similarities at once
S = Q @ D.T              # (N, M)

# Time: O(N × M × d) but with BLAS optimization
```

**Speedup:** 10-100x depending on N, M

---

### **Approximate Nearest Neighbor (ANN)**

For large-scale search, exact similarity is too slow.

#### **HNSW (Hierarchical Navigable Small World)**

**Complexity:**
- **Build:** O(N log N × d)
- **Query:** O(log N × d)

**vs Exact:**
- **Build:** O(N × d)
- **Query:** O(N × d)

**Trade-off:**
- **Recall:** 95-99% (slightly lower than exact)
- **Speed:** 10-1000x faster

**Parameters:**
```python
{
    "m": 16,               # Number of links per node
    "ef_construction": 200,  # Search depth during build
    "ef_search": 100        # Search depth during query
}
```

---

## 📊 Empirical Analysis

### **Distance Distribution**

For random vectors in high dimensions:

**Theorem:** Cosine similarity → Gaussian(0, 1/√d)

```python
# Generate random vectors
d = 768
N = 10000

random_vecs = np.random.randn(N, d)
random_vecs = random_vecs / np.linalg.norm(random_vecs, axis=1, keepdims=True)

# Compute pairwise similarities
sims = random_vecs @ random_vecs.T

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(sims.flatten(), bins=100)
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Random Vector Similarities")

# Result: Bell curve centered at 0
# Mean ≈ 0.0
# Std ≈ 1/sqrt(768) ≈ 0.036
```

**Implication:**
- Random vectors are nearly orthogonal
- Similarity > 0.1 is already meaningful
- Similarity > 0.5 is very high

---

### **Embedding Space Analysis**

**BERT Embeddings (768-dim):**

```
Typical ranges:
- Same sentence:         0.7 - 1.0
- Paraphrases:          0.5 - 0.7
- Related topics:       0.3 - 0.5
- Different topics:     0.0 - 0.3
- Opposite meaning:    -0.2 - 0.0
```

**BGE Embeddings (1024-dim):**

```
Higher quality separation:
- Same sentence:         0.8 - 1.0
- Paraphrases:          0.6 - 0.8
- Related topics:       0.4 - 0.6
- Different topics:     0.1 - 0.4
- Opposite meaning:    -0.1 - 0.1
```

---

## 🎯 Practical Guidelines

### **Choosing a Metric**

| Use Case | Best Metric | Why |
|----------|-------------|-----|
| **Text embeddings** | Cosine | Direction matters, not magnitude |
| **Image features** | Cosine or L2 | Both work well |
| **Product recommendations** | Cosine | User preferences are directional |
| **Clustering** | L2 (Euclidean) | Natural geometric interpretation |
| **Anomaly detection** | Mahalanobis | Accounts for correlations |

### **Threshold Selection**

**For Cosine Similarity:**

```python
SIMILARITY_THRESHOLDS = {
    "very_strict": 0.8,   # Only near-duplicates
    "strict": 0.5,        # Strong semantic match
    "normal": 0.3,        # Related content
    "lenient": 0.1,       # Weak relation
}
```

**RAGFlow Default:** 0.2 (between lenient and normal)

---

## 🧮 Mathematical Proofs

### **Proof: Cosine = Dot Product for Normalized Vectors**

Given: $||\mathbf{a}|| = ||\mathbf{b}|| = 1$

$$
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||} = \frac{\mathbf{a} \cdot \mathbf{b}}{1 \cdot 1} = \mathbf{a} \cdot \mathbf{b} \quad \square
$$

### **Proof: Triangle Inequality**

For any vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$:

$$
d(\mathbf{a}, \mathbf{c}) \leq d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c})
$$

**Proof:**

$$
\begin{align}
d(\mathbf{a}, \mathbf{c}) &= ||\mathbf{a} - \mathbf{c}|| \\
&= ||(\mathbf{a} - \mathbf{b}) + (\mathbf{b} - \mathbf{c})|| \\
&\leq ||\mathbf{a} - \mathbf{b}|| + ||\mathbf{b} - \mathbf{c}|| \\
&= d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c}) \quad \square
\end{align}
$$

---

## 🔧 Implementation in RAGFlow

### **Elasticsearch Script**

```python
# Used in rag/utils/es_conn.py
script = {
    "source": "cosineSimilarity(params.query_vector, 'q_vec') + 1.0",
    "params": {
        "query_vector": query_vec.tolist()
    }
}
```

**Why +1.0?**
- Ensures all scores are positive
- Range becomes [0, 2] instead of [-1, 1]
- Easier to combine with BM25 scores

### **Embedding Model**

```python
# rag/llm/embedding_model.py
class EmbeddingModel:
    def encode(self, texts, normalize=True):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize  # L2 normalize
        )
        return embeddings
```

---

## 📝 Summary

### **Key Formulas**

| Metric | Formula | Range | Normalized? |
|--------|---------|-------|-------------|
| **Cosine** | $\frac{\mathbf{a} \cdot \mathbf{b}}{\\|\\|\mathbf{a}\\|\\| \\|\\|\mathbf{b}\\|\\|}$ | [-1, 1] | Yes |
| **Dot Product** | $\sum a_i b_i$ | (-∞, ∞) | No |
| **Euclidean** | $\sqrt{\sum (a_i - b_i)^2}$ | [0, ∞) | No |
| **Manhattan** | $\sum \\|a_i - b_i\\|$ | [0, ∞) | No |

### **Performance Tips**

✅ Use numpy vectorization
✅ Normalize vectors once, reuse
✅ Batch operations when possible
✅ Use ANN for large-scale search
✅ Cache frequently used embeddings

### **Common Mistakes**

❌ Forgetting to normalize
❌ Using wrong metric for use case
❌ Not handling zero vectors
❌ Inefficient loops
❌ Recomputing same embeddings

---

**Related Files:**
- [hybrid_search_algorithm.md](../03-RAG-ENGINE/hybrid_search_algorithm.md)
- [embedding_generation.md](../03-RAG-ENGINE/embedding_generation.md)

**Last Updated:** 2025-11-23
