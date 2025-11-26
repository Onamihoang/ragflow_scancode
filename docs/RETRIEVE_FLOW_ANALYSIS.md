# Phân Tích Luồng Retrieve của RAGFlow

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Sơ đồ luồng tổng thể](#2-sơ-đồ-luồng-tổng-thể)
3. [Chi tiết từng giai đoạn](#3-chi-tiết-từng-giai-đoạn)
4. [Các thành phần chính](#4-các-thành-phần-chính)
5. [Thuật toán scoring và ranking](#5-thuật-toán-scoring-và-ranking)
6. [Cấu hình và tham số](#6-cấu-hình-và-tham-số)
7. [Ví dụ minh họa](#7-ví-dụ-minh-họa)

---

## 1. Tổng quan

RAGFlow sử dụng phương pháp **Hybrid Search** kết hợp giữa:
- **BM25 (Keyword Search)**: Tìm kiếm dựa trên từ khóa, đo lường sự trùng khớp về mặt văn bản
- **Vector Search (Semantic Search)**: Tìm kiếm dựa trên ngữ nghĩa thông qua embedding vectors

Luồng retrieve được thiết kế theo kiến trúc **pipeline** với nhiều giai đoạn xử lý, từ tiền xử lý câu hỏi đến reranking và trích dẫn kết quả.

---

## 2. Sơ đồ luồng tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
│                   "RAGFlow hoạt động như thế nào?"                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 1: TIỀN XỬ LÝ CÂU HỎI                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  Multi-turn     │  │  Cross-language │  │  Keyword            │  │
│  │  Refinement     │  │  Translation    │  │  Extraction         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 2: MÃ HÓA TRUY VẤN                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Embedding Model: Query → Vector [0.12, -0.45, 0.78, ...]      │ │
│  │  Fulltext Parser: Query → BM25 Expression (keywords + weights) │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 3: HYBRID SEARCH                       │
│  ┌─────────────────────────┐    ┌─────────────────────────┐         │
│  │    BM25 SEARCH          │    │    VECTOR SEARCH        │         │
│  │    (Keyword Matching)   │    │    (Semantic Matching)  │         │
│  │    Weight: 5%           │    │    Weight: 95%          │         │
│  └─────────────────────────┘    └─────────────────────────┘         │
│                    │                        │                        │
│                    └──────────┬─────────────┘                        │
│                               ▼                                      │
│                    ┌─────────────────────────┐                       │
│                    │  WEIGHTED SUM FUSION    │                       │
│                    │  (Kết hợp điểm số)      │                       │
│                    └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 4: DOCUMENT STORE QUERY                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Elasticsearch / Infinity / OpenSearch                         │ │
│  │  - Index: ragflow_{tenant_id}                                  │ │
│  │  - Filters: kb_ids, doc_ids, available_int=1                   │ │
│  │  - Return: Top 1024 chunks (mặc định)                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 5: RERANKING                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Option A: Model-based Reranking (nếu có rerank model)      │    │
│  │  - Sử dụng LLM/cross-encoder để tính relevance score        │    │
│  │                                                              │    │
│  │  Option B: Hybrid Reranking (mặc định)                      │    │
│  │  - Token similarity: 70%                                     │    │
│  │  - Vector similarity: 30%                                    │    │
│  │  - Rank features (PageRank): +bonus                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 6: LỌC VÀ PHÂN TRANG                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  - Lọc theo similarity_threshold (mặc định ≥ 0.1)              │ │
│  │  - Sắp xếp giảm dần theo điểm similarity                       │ │
│  │  - Lấy top_n kết quả (mặc định: 6 chunks)                      │ │
│  │  - Aggregate theo document (nếu cần)                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 7: TĂNG CƯỜNG (TÙY CHỌN)               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  Knowledge      │  │  Table of       │  │  Metadata           │  │
│  │  Graph (KG)     │  │  Contents (TOC) │  │  Filtering          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 8: FORMAT & CITATION                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  - Format chunks thành knowledge pieces cho LLM                │ │
│  │  - Truncate để fit vào context window                          │ │
│  │  - Chèn citations [ID:1], [ID:2] vào câu trả lời               │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RETRIEVED CHUNKS                            │
│                   + Citations + Document References                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết từng giai đoạn

### 3.1. Giai đoạn 1: Tiền xử lý câu hỏi (Query Preprocessing)

#### Mục đích
Biến đổi câu hỏi của người dùng thành dạng tối ưu cho việc tìm kiếm.

#### Các bước xử lý

**a) Multi-turn Refinement (Tinh chỉnh hội thoại đa lượt)**
- **Khi nào**: Khi người dùng có nhiều câu hỏi liên tiếp trong một cuộc hội thoại
- **Logic**:
  - Lấy 3 câu hỏi gần nhất từ lịch sử hội thoại
  - Nếu bật tính năng `refine_multiturn`: Sử dụng LLM để tổng hợp các câu hỏi thành một câu hỏi hoàn chỉnh
  - Ví dụ: "Nó hoạt động như thế nào?" → "RAGFlow hoạt động như thế nào?" (dựa trên context trước đó)

**b) Cross-language Translation (Dịch đa ngôn ngữ)**
- **Khi nào**: Khi tài liệu được lưu trữ bằng nhiều ngôn ngữ
- **Logic**:
  - Dịch câu hỏi sang các ngôn ngữ mục tiêu được cấu hình
  - Tìm kiếm song song trên tất cả các ngôn ngữ
  - Ví dụ: "What is RAGFlow?" → ["What is RAGFlow?", "RAGFlow是什么?", "RAGFlow là gì?"]

**c) Keyword Extraction (Trích xuất từ khóa)**
- **Khi nào**: Khi bật tính năng `keyword`
- **Logic**:
  - Sử dụng LLM để trích xuất các từ khóa quan trọng từ câu hỏi
  - Nối thêm các từ khóa vào câu hỏi gốc để tăng cường tìm kiếm
  - Ví dụ: "Cách cài đặt RAGFlow?" + keywords: "installation, setup, docker, requirements"

---

### 3.2. Giai đoạn 2: Mã hóa truy vấn (Query Encoding)

#### Mục đích
Chuyển đổi câu hỏi văn bản thành các biểu diễn có thể tìm kiếm được.

#### Hai luồng xử lý song song

**a) Vector Encoding (Embedding)**
```
Input:  "RAGFlow hoạt động như thế nào?"
        ↓
        Embedding Model (OpenAI, Cohere, HuggingFace, etc.)
        ↓
Output: [0.12, -0.45, 0.78, ..., 0.23]  (384-1024 dimensions)
```

- Sử dụng embedding model được cấu hình (OpenAI, Jina, BGE, etc.)
- Tạo ra vector dense với số chiều tùy thuộc vào model
- Vector này sẽ được so sánh với vectors của các chunks đã lưu trữ

**b) Fulltext Query Parsing (BM25)**
```
Input:  "RAGFlow hoạt động như thế nào?"
        ↓
        FulltextQueryer.question()
        ↓
Output: {
          "must": ["ragflow", "hoạt", "động"],
          "should": ["như", "thế", "nào"],
          "weights": {
            "title": 10x,
            "important_kwd": 30x,
            "content": 2x
          }
        }
```

- Tokenize câu hỏi thành các từ/từ ghép
- Loại bỏ stopwords (từ không quan trọng)
- Xác định trọng số cho từng field trong document
- Xử lý đặc biệt cho tiếng Trung, tiếng Việt (word segmentation)

---

### 3.3. Giai đoạn 3: Hybrid Search

#### Mục đích
Kết hợp hai phương pháp tìm kiếm để tận dụng ưu điểm của cả hai.

#### Logic kết hợp

```
                    ┌──────────────────────────────┐
                    │       HYBRID SEARCH          │
                    └──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                      ▼
    ┌───────────────────┐                ┌───────────────────┐
    │    BM25 SEARCH    │                │   VECTOR SEARCH   │
    │                   │                │                   │
    │ • Exact matching  │                │ • Semantic match  │
    │ • Keyword focus   │                │ • Context-aware   │
    │ • Good for names  │                │ • Good for meaning│
    │ • Fast            │                │ • Handle synonyms │
    │                   │                │                   │
    │ Weight: 5%        │                │ Weight: 95%       │
    └───────────────────┘                └───────────────────┘
            │                                      │
            └──────────────────┬───────────────────┘
                               ▼
                    ┌──────────────────────────────┐
                    │     WEIGHTED SUM FUSION      │
                    │                              │
                    │ final_score = 0.05 × BM25    │
                    │             + 0.95 × Vector  │
                    └──────────────────────────────┘
```

#### Ưu điểm của từng phương pháp

| Tiêu chí | BM25 (Keyword) | Vector (Semantic) |
|----------|----------------|-------------------|
| **Tìm kiếm tên riêng** | ✅ Rất tốt | ❌ Có thể miss |
| **Tìm kiếm khái niệm** | ❌ Cần exact match | ✅ Hiểu ngữ nghĩa |
| **Xử lý từ đồng nghĩa** | ❌ Không | ✅ Tự động |
| **Tốc độ** | ✅ Rất nhanh | ⚡ Nhanh (với ANN) |
| **Giải thích được** | ✅ Dễ debug | ❌ Black box |

---

### 3.4. Giai đoạn 4: Document Store Query

#### Mục đích
Thực thi truy vấn trên kho lưu trữ tài liệu.

#### Các Document Store được hỗ trợ

| Engine | Đặc điểm |
|--------|----------|
| **Elasticsearch** | Mặc định, hybrid search tốt, cần normalize scores |
| **Infinity** | Scores đã normalized sẵn, tối ưu cho RAG |
| **OpenSearch** | Tương tự Elasticsearch |
| **OceanBase** | SQL-based, cho môi trường doanh nghiệp |

#### Cấu trúc Index

```
Index: ragflow_{tenant_id}
├── content_ltks     (text)     - Nội dung đã tokenize
├── title_tks        (text)     - Tiêu đề đã tokenize
├── important_kwd    (keyword)  - Từ khóa quan trọng
├── q_{dim}_vec      (vector)   - Dense vector embedding
├── docnm_kwd        (keyword)  - Tên document
├── doc_id           (keyword)  - ID document
├── kb_id            (keyword)  - ID knowledge base
├── chunk_id         (keyword)  - ID chunk
├── available_int    (integer)  - Trạng thái khả dụng (1=active)
└── [metadata fields]           - Các trường metadata tùy chỉnh
```

#### Quá trình truy vấn

1. **Xây dựng query compound**:
   - MatchTextExpr: BM25 query cho full-text search
   - MatchDenseExpr: Vector similarity query
   - FusionExpr: Kết hợp scores

2. **Áp dụng filters**:
   - kb_id IN [danh sách knowledge bases được phép]
   - doc_id IN [danh sách documents được lọc] (nếu có metadata filter)
   - available_int = 1 (chỉ lấy chunks active)

3. **Pagination**:
   - Lấy tối đa `top_k` kết quả (mặc định 1024)
   - Batch size cho reranking: RERANK_LIMIT (64)

---

### 3.5. Giai đoạn 5: Reranking

#### Mục đích
Sắp xếp lại kết quả để đưa các chunks liên quan nhất lên đầu.

#### Hai chế độ Reranking

**Chế độ A: Model-based Reranking (Nếu có cấu hình rerank model)**

```
Query: "RAGFlow là gì?"
Chunk 1: "RAGFlow is an open-source RAG engine..."
Chunk 2: "The installation process requires Docker..."
        ↓
    Rerank Model (Jina, Cohere, BGE-reranker, etc.)
        ↓
Scores: [0.95, 0.23]  (relevance scores 0-1)
```

- Sử dụng cross-encoder hoặc LLM-based reranker
- Tính toán relevance score trực tiếp giữa query và mỗi chunk
- Chính xác hơn nhưng chậm hơn

**Chế độ B: Hybrid Reranking (Mặc định, không cần rerank model)**

```
final_score = tkweight × token_similarity
            + vtweight × vector_similarity
            + rank_features

Trong đó:
- tkweight = 0.7 (70% - token/keyword similarity)
- vtweight = 0.3 (30% - vector/semantic similarity)
- rank_features = PageRank bonus + label bonus
```

#### Công thức chi tiết

```
Token Similarity (tsim):
- Đếm số từ trùng khớp giữa query và chunk
- Normalize theo độ dài
- Áp dụng IDF weighting

Vector Similarity (vsim):
- Cosine similarity giữa query vector và chunk vector
- Đã được tính từ giai đoạn search

Rank Features:
- PageRank: Điểm quan trọng của document gốc
- Labels: Bonus cho các chunks có tags liên quan
```

---

### 3.6. Giai đoạn 6: Lọc và phân trang

#### Quá trình lọc

```python
# 1. Chuyển scores thành numpy array để xử lý nhanh
scores = np.array(similarity_scores)

# 2. Sắp xếp giảm dần theo điểm
sorted_indices = np.argsort(scores * -1)

# 3. Lọc theo ngưỡng similarity
valid_indices = [i for i in sorted_indices
                 if scores[i] >= similarity_threshold]

# 4. Phân trang
page_indices = valid_indices[(page-1)*page_size : page*page_size]
```

#### Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `similarity_threshold` | 0.1 | Ngưỡng tối thiểu để giữ chunk |
| `top_n` | 6 | Số chunks trả về cho LLM |
| `top_k` | 1024 | Số chunks xem xét trước khi lọc |
| `page` | 1 | Trang hiện tại |
| `page_size` | = top_n | Số items mỗi trang |

#### Aggregation theo Document

Nếu cần, hệ thống cũng aggregate kết quả theo document:

```
doc_aggs = [
  {"doc_name": "RAGFlow_Guide.pdf", "doc_id": "xxx", "count": 3},
  {"doc_name": "Installation.md", "doc_id": "yyy", "count": 2},
  {"doc_name": "FAQ.docx", "doc_id": "zzz", "count": 1}
]
```

---

### 3.7. Giai đoạn 7: Tăng cường (Enhancement)

#### 7a. Knowledge Graph Retrieval (Tùy chọn)

```
Khi bật use_kg=True:
        ↓
Query → Knowledge Graph Query
        ↓
Trích xuất entities và relationships liên quan
        ↓
Thêm vào đầu danh sách chunks
```

- Bổ sung thông tin cấu trúc từ knowledge graph
- Đặc biệt hữu ích cho các câu hỏi về mối quan hệ

#### 7b. Table of Contents Enhancement (Tùy chọn)

```
Khi bật toc_enhance=True:
        ↓
Phân tích cấu trúc TOC của documents
        ↓
Tìm các sections liên quan dựa trên heading hierarchy
        ↓
Thêm context từ parent/sibling sections
```

- Cải thiện context cho các câu hỏi liên quan đến cấu trúc tài liệu

#### 7c. Metadata Filtering

```
Nếu có metadata_filter:
        ↓
Lọc documents theo metadata rules:
- Theo loại file (PDF, DOCX, etc.)
- Theo ngày tạo/cập nhật
- Theo tags/categories
- Theo custom fields
```

---

### 3.8. Giai đoạn 8: Format và Citation

#### Formatting cho LLM

```
Input chunks:
[
  {"content": "RAGFlow is an open-source...", "similarity": 0.95},
  {"content": "Installation requires...", "similarity": 0.87}
]
        ↓
Knowledge string:
"""
1. RAGFlow is an open-source RAG engine based on deep document understanding.
   It provides high-quality retrieval with citations.

2. Installation requires Docker and docker-compose. The system needs at least
   16GB RAM and 50GB disk space.
"""
```

- Truncate nếu vượt quá token limit của LLM
- Format theo template được cấu hình

#### Citation Insertion

```
LLM Response: "RAGFlow is an open-source RAG engine that provides..."
        ↓
Citation Matching:
- So sánh mỗi câu trong response với các chunks
- Tính similarity (token + vector)
- Nếu similarity > threshold → chèn citation
        ↓
Final Response: "RAGFlow is an open-source RAG engine [1] that provides..."
```

---

## 4. Các thành phần chính

### 4.1. Bảng tổng hợp các thành phần

| Thành phần | Chức năng | Files chính |
|------------|-----------|-------------|
| **API Layer** | Xử lý HTTP requests | `api/apps/conversation_app.py` |
| **Dialog Service** | Điều phối toàn bộ luồng | `api/db/services/dialog_service.py` |
| **Search Dealer** | Engine tìm kiếm hybrid | `rag/nlp/search.py` |
| **Query Parser** | Parse và tokenize query | `rag/nlp/query.py` |
| **Embedding Models** | Encode text → vector | `rag/llm/embedding_model.py` |
| **Rerank Models** | Reranking results | `rag/llm/rerank_model.py` |
| **Document Store** | Lưu trữ và index | `rag/utils/es_conn.py`, `infinity_conn.py` |
| **Agent Retrieval** | Component cho agent canvas | `agent/tools/retrieval.py` |

### 4.2. Mô hình Class Diagram (Logic)

```
┌─────────────────────────────────────────────────────────────────┐
│                     ConversationApp (API)                       │
│  ─────────────────────────────────────────────────────────────  │
│  + completion(messages, dialog_id) → Stream[Response]           │
│  + ask(question, kb_ids) → Response                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DialogService                               │
│  ─────────────────────────────────────────────────────────────  │
│  + chat(dialog, messages, stream) → Generator[Answer]           │
│  + ask(kb_ids, question) → Answer                               │
│  - get_models(dialog) → (embedding, chat, rerank)               │
│  - format_knowledge(chunks, max_tokens) → string                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Dealer (Search Engine)                      │
│  ─────────────────────────────────────────────────────────────  │
│  + retrieval(question, embd_mdl, kb_ids, ...) → KBInfos         │
│  + search(query, kb_ids, embd_mdl, ...) → RawResults            │
│  + rerank(results, question, ...) → ScoredResults               │
│  + rerank_by_model(model, results, question) → ScoredResults    │
│  + insert_citations(answer, chunks) → AnswerWithCitations       │
└─────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  EmbeddingModel  │  │  FulltextQueryer │  │  RerankModel     │
│  ──────────────  │  │  ──────────────  │  │  ──────────────  │
│  + encode()      │  │  + question()    │  │  + similarity()  │
│  + encode_queries│  │  + keywords()    │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               DocStoreConnection (ES/Infinity)                  │
│  ─────────────────────────────────────────────────────────────  │
│  + search(index, query, filters, top_k) → Documents             │
│  + get(index, doc_ids) → Documents                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Thuật toán Scoring và Ranking

### 5.1. BM25 Score

```
BM25(Q, D) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))

Trong đó:
- Q = Query (tập các từ khóa)
- D = Document (chunk)
- qi = từ khóa thứ i trong query
- f(qi, D) = tần suất xuất hiện của qi trong D
- |D| = độ dài của D
- avgdl = độ dài trung bình của tất cả documents
- k1 = 1.2 (điều chỉnh saturation)
- b = 0.75 (điều chỉnh length normalization)
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
```

### 5.2. Vector Similarity (Cosine)

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Trong đó:
- A = query vector
- B = chunk vector
- A · B = dot product
- ||A||, ||B|| = vector magnitudes
```

### 5.3. Hybrid Score Fusion

```
# Trong quá trình search (trước reranking)
fusion_score = 0.05 × normalized_bm25 + 0.95 × vector_similarity

# Trong quá trình reranking (hybrid mode)
final_score = (1 - vector_weight) × token_similarity
            + vector_weight × vector_similarity
            + rank_features

# Với model-based reranking
final_score = rerank_model.similarity(query, chunk)  # 0-1 range
```

### 5.4. Token Similarity (Trong Reranking)

```
token_sim = Σ (1 if token in chunk_tokens else 0) / len(query_tokens)

# Với IDF weighting
token_sim_idf = Σ (IDF(token) if token in chunk_tokens else 0) / Σ IDF(query_tokens)
```

---

## 6. Cấu hình và tham số

### 6.1. Dialog Configuration

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `top_n` | int | 6 | Số chunks trả về cho LLM |
| `top_k` | int | 1024 | Số chunks xem xét trước khi lọc |
| `similarity_threshold` | float | 0.1 | Ngưỡng similarity tối thiểu |
| `vector_similarity_weight` | float | 0.3 | Tỉ trọng vector trong hybrid (0.3 = 30% vector, 70% keyword) |
| `rerank_id` | string | "" | ID của rerank model (empty = không dùng) |
| `kb_ids` | list[str] | [] | Danh sách Knowledge Base IDs để tìm kiếm |

### 6.2. Prompt Configuration

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `refine_multiturn` | bool | false | Gộp multi-turn questions thành 1 |
| `cross_languages` | list[str] | [] | Ngôn ngữ dịch thêm (e.g., ["en", "zh"]) |
| `keyword` | bool | false | Trích xuất keywords bổ sung |
| `toc_enhance` | bool | false | Tăng cường với Table of Contents |
| `use_kg` | bool | false | Sử dụng Knowledge Graph |
| `system` | string | template | System prompt với {knowledge} placeholder |

### 6.3. Metadata Filter

```json
{
  "meta_data_filter": {
    "mode": "auto",  // hoặc "manual"
    "rules": [
      {"field": "file_type", "operator": "in", "value": ["pdf", "docx"]},
      {"field": "created_at", "operator": ">=", "value": "2024-01-01"}
    ]
  }
}
```

---

## 7. Ví dụ minh họa

### 7.1. Ví dụ: Tìm kiếm đơn giản

**Input**:
```
User: "RAGFlow là gì?"
Dialog config: {top_n: 3, similarity_threshold: 0.2}
```

**Quá trình**:

```
Step 1: Tiền xử lý
├── Query gốc: "RAGFlow là gì?"
├── Multi-turn: Không (câu hỏi đầu tiên)
└── Output: ["RAGFlow là gì?"]

Step 2: Mã hóa
├── Vector: [0.12, -0.45, ...] (384 dims)
└── BM25: {"must": ["ragflow"], "should": ["là", "gì"]}

Step 3: Hybrid Search
├── BM25 Results: [chunk_5, chunk_12, chunk_3, ...]
├── Vector Results: [chunk_5, chunk_8, chunk_12, ...]
└── Fused: [chunk_5 (0.92), chunk_12 (0.85), chunk_8 (0.78), ...]

Step 4: Document Store Query
└── Retrieved 156 chunks from Elasticsearch

Step 5: Reranking (Hybrid mode)
├── Token similarity + Vector similarity
└── Sorted: [chunk_5 (0.91), chunk_12 (0.84), chunk_3 (0.72), ...]

Step 6: Filtering
├── Threshold 0.2: 45 chunks pass
├── Top 3: [chunk_5, chunk_12, chunk_3]
└── Aggregated: [{doc: "README.md", count: 2}, {doc: "docs/intro.md", count: 1}]

Step 7: Format
└── Knowledge: "1. RAGFlow is an open-source RAG engine...
                2. Deep document understanding...
                3. Supports multiple file formats..."

Output:
{
  "chunks": [
    {"content": "RAGFlow is an open-source...", "similarity": 0.91, "doc": "README.md"},
    {"content": "Deep document understanding...", "similarity": 0.84, "doc": "README.md"},
    {"content": "Supports PDF, DOCX...", "similarity": 0.72, "doc": "docs/intro.md"}
  ],
  "doc_aggs": [
    {"doc_name": "README.md", "count": 2},
    {"doc_name": "docs/intro.md", "count": 1}
  ]
}
```

### 7.2. Ví dụ: Multi-turn với Reranking

**Input**:
```
Message 1: "Giới thiệu về RAGFlow"
Message 2: "Nó hỗ trợ những định dạng file nào?"
Message 3: "Còn về hiệu suất thì sao?"

Dialog config: {
  top_n: 5,
  refine_multiturn: true,
  rerank_id: "jina-reranker-v2"
}
```

**Quá trình**:

```
Step 1: Multi-turn Refinement
├── Last 3 questions: ["Giới thiệu về RAGFlow", "Nó hỗ trợ những định dạng file nào?", "Còn về hiệu suất thì sao?"]
├── LLM Refinement: "Hiệu suất của RAGFlow là như thế nào?"
└── Expanded query: "Hiệu suất performance benchmark RAGFlow RAG engine"

Step 2-4: [Tương tự như trên]

Step 5: Model-based Reranking
├── Rerank Model: Jina Reranker v2
├── Input: [(query, chunk_1), (query, chunk_2), ...]
├── Scores: [0.95, 0.87, 0.82, 0.65, 0.45, ...]
└── Sorted by rerank scores

Step 6-7: [Tương tự như trên]
```

---

## Kết luận

Luồng retrieve của RAGFlow được thiết kế với các đặc điểm nổi bật:

1. **Hybrid Search**: Kết hợp BM25 và Vector search để tận dụng ưu điểm của cả hai
2. **Flexible Reranking**: Hỗ trợ cả hybrid scoring và model-based reranking
3. **Multi-turn Support**: Có thể tổng hợp ngữ cảnh từ nhiều lượt hội thoại
4. **Enhancement Options**: Knowledge Graph, TOC, Metadata filtering
5. **Citation System**: Tự động chèn trích dẫn vào câu trả lời
6. **Configurable**: Nhiều tham số có thể điều chỉnh theo use case

Thiết kế này cho phép RAGFlow xử lý hiệu quả các tình huống tìm kiếm đa dạng, từ exact matching đến semantic understanding.
