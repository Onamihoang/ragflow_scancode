# RAG Engine Tech Stack Analysis

## Tổng Quan

RAGFlow là một open-source RAG (Retrieval-Augmented Generation) engine dựa trên deep document understanding. Module RAG engine là core component xử lý toàn bộ pipeline từ document parsing đến retrieval và generation.

---

## 1. Document Processing & Parsing

### 1.1 PDF Processing
| Library | Version | Mục đích |
|---------|---------|----------|
| **pdfplumber** | 0.10.4 | Trích xuất text, characters, và metadata từ PDF |
| **pypdf** | 6.0.0 | Đọc PDF structure, outlines, và basic text extraction |
| **pypdf2** | 3.0.1 | Legacy PDF reading support |

**Chi tiết implementation:**
- `deepdoc/parser/pdf_parser.py`: Class `RAGFlowPdfParser` sử dụng pdfplumber để convert PDF thành images và extract characters
- Hỗ trợ multi-page processing với async/await pattern sử dụng **trio**
- Tích hợp layout analysis và table structure recognition

### 1.2 OCR (Optical Character Recognition)
| Library | Version | Mục đích |
|---------|---------|----------|
| **onnxruntime** | 1.19.2 | Chạy OCR models (CPU) |
| **onnxruntime-gpu** | 1.19.2 | GPU acceleration cho OCR |
| **opencv-python** | 4.10.0.84 | Image preprocessing và manipulation |
| **numpy** | <2.0.0 | Array operations cho image data |

**Chi tiết implementation:**
- `deepdoc/vision/ocr.py`:
  - `TextDetector`: Phát hiện vùng chứa text trong image
  - `TextRecognizer`: Nhận dạng text từ detected regions
  - Sử dụng ONNX models từ HuggingFace (`InfiniFlow/deepdoc`)
  - Hỗ trợ multi-GPU processing với `PARALLEL_DEVICES`

### 1.3 Layout Recognition
| Library | Mục đích |
|---------|----------|
| **LayoutRecognizer** | Nhận dạng layout elements (tables, figures, text blocks) |
| **AscendLayoutRecognizer** | Huawei Ascend NPU support |
| **TableStructureRecognizer** | Nhận dạng cấu trúc bảng (rows, columns, headers) |

### 1.4 Document Format Support
| Library | Version | Formats |
|---------|---------|---------|
| **python-docx** | >=1.1.2 | .docx files |
| **python-pptx** | >=1.0.2 | .pptx files |
| **openpyxl** | >=3.1.0 | .xlsx files |
| **extract-msg** | >=0.39.0 | Outlook .msg files |
| **tika** | 2.6.0 | Multiple formats via Apache Tika |
| **mammoth** | >=1.11.0 | Word documents to HTML |

---

## 2. NLP & Text Processing

### 2.1 Tokenization & Text Analysis
| Library | Version | Mục đích |
|---------|---------|----------|
| **datrie** | >=0.8.3 | Fast trie-based tokenization cho Chinese |
| **nltk** | 3.9.1 | Natural language processing toolkit |
| **tiktoken** | 0.7.0 | OpenAI's tokenizer cho token counting |
| **cn2an** | 0.5.22 | Chinese number conversion |
| **hanziconv** | 0.3.2 | Chinese character conversion |
| **xpinyin** | 0.7.6 | Chinese pinyin conversion |

**Chi tiết implementation:**
- `rag/nlp/rag_tokenizer.py`: Custom tokenizer hỗ trợ:
  - Fine-grained tokenization cho cả English và Chinese
  - POS tagging
  - Language detection

### 2.2 Machine Learning cho Text Processing
| Library | Version | Mục đích |
|---------|---------|----------|
| **xgboost** | 1.6.0 | Text concatenation prediction model |
| **scikit-learn** | 1.5.0 | Clustering (KMeans), metrics (silhouette_score) |
| **umap_learn** | 0.5.6 | Dimensionality reduction |

---

## 3. LLM Integration

### 3.1 LLM Providers
| Provider | Library | Version |
|----------|---------|---------|
| **OpenAI** | openai | >=1.45.0 |
| **Anthropic** | anthropic | 0.34.1 |
| **Google** | google-generativeai, vertexai | >=0.8.1, 1.70.0 |
| **Cohere** | cohere | 5.6.2 |
| **Mistral** | mistralai | 0.4.2 |
| **Groq** | groq | 0.9.0 |
| **Ollama** | ollama | >=0.5.0 |
| **ZhipuAI** | zhipuai | 2.0.1 |
| **Dashscope (Alibaba)** | dashscope | 1.20.11 |
| **Qianfan (Baidu)** | qianfan | 0.4.6 |
| **Volcengine** | volcengine | 1.0.194 |
| **Replicate** | replicate | 0.31.0 |
| **VoyageAI** | voyageai | 0.2.3 |
| **LiteLLM** | litellm | >=1.74.15 |

**Chi tiết implementation:**
- `rag/llm/chat_model.py`: Base class cho tất cả chat models
- `rag/llm/embedding_model.py`: Embedding model abstraction
- `rag/llm/rerank_model.py`: Reranking model abstraction
- Hỗ trợ streaming responses và async operations

### 3.2 Embedding Models
Hỗ trợ embedding từ nhiều providers:
- OpenAI embeddings
- Hugging Face models (via `infinity-emb`)
- Local ONNX models
- Cohere embeddings
- VoyageAI embeddings

---

## 4. Vector Search & Document Store

### 4.1 Primary Search Engines
| Engine | Library | Version | Mục đích |
|--------|---------|---------|----------|
| **Elasticsearch** | elasticsearch | 8.12.1 | Full-text search & vector search |
| **Elasticsearch DSL** | elasticsearch-dsl | 8.12.0 | Query building |
| **Infinity** | infinity-sdk | 0.6.6 | Alternative vector database |
| **OpenSearch** | opensearch-py | 2.7.1 | OpenSearch support |

**Chi tiết implementation:**
- `rag/utils/es_conn.py`: Elasticsearch connection với:
  - Hybrid search (text + vector)
  - KNN queries
  - Aggregations
  - Bulk operations
- `rag/utils/infinity_conn.py`: Infinity database connection
- Hỗ trợ chuyển đổi giữa Elasticsearch và Infinity qua `DOC_ENGINE` config

### 4.2 Hybrid Search Features
- **Text Search**: BM25-based full-text search
- **Dense Vector Search**: KNN với cosine similarity
- **Fusion**: Weighted sum của text và vector scores
- **Rank Features**: PageRank và custom scoring

---

## 5. Caching & Message Queue

### 5.1 Redis/Valkey
| Library | Version | Mục đích |
|---------|---------|----------|
| **valkey** | 6.0.2 | Redis-compatible caching & message queue |

**Chi tiết implementation:**
- `rag/utils/redis_conn.py`:
  - Session caching
  - Distributed locks (`RedisDistributedLock`)
  - Message queues (Redis Streams)
  - Lua scripts cho atomic operations

---

## 6. Object Storage

### 6.1 Storage Backends
| Library | Version | Mục đích |
|---------|---------|----------|
| **minio** | 7.2.4 | S3-compatible object storage |
| **boto3** | 1.34.140 | AWS S3 SDK |
| **azure-storage-blob** | 12.22.0 | Azure Blob Storage |
| **opendal** | >=0.45.0 | Unified data access layer |

**Chi tiết implementation:**
- `rag/utils/minio_conn.py`: MinIO connection cho document storage
- Hỗ trợ multi-cloud storage

---

## 7. Graph RAG

### 7.1 Knowledge Graph
| Library | Version | Mục đích |
|---------|---------|----------|
| **networkx** | (bundled) | Graph data structure |
| **graspologic** | custom fork | Graph algorithms |

**Chi tiết implementation:**
- `graphrag/general/graph_extractor.py`:
  - Entity extraction từ text sử dụng LLM
  - Relation extraction
  - Graph construction với NetworkX
  - Community detection

### 7.2 Graph Algorithms
- Entity và relationship extraction
- Graph clustering
- Community summarization
- Graph-based retrieval

---

## 8. Async & Concurrency

### 8.1 Async Framework
| Library | Version | Mục đích |
|---------|---------|----------|
| **trio** | >=0.17.0 | Structured concurrency |
| **Quart** | 0.20.0 | Async web framework |

**Chi tiết implementation:**
- Parallel OCR processing với `trio.open_nursery()`
- Async LLM calls
- Rate limiting với `CapacityLimiter`

---

## 9. Web Framework & API

### 9.1 Backend Framework
| Library | Version | Mục đích |
|---------|---------|----------|
| **Flask** | 3.0.3 | Sync web framework |
| **Quart** | 0.20.0 | Async web framework |
| **flask-cors** | 5.0.0 | CORS handling |
| **flask-login** | 0.6.3 | Authentication |
| **flasgger** | >=0.9.7.1 | Swagger documentation |

---

## 10. Database ORM

| Library | Version | Mục đích |
|---------|---------|----------|
| **peewee** | 3.17.1 | ORM cho MySQL |
| **pymysql** | >=1.1.1 | MySQL driver |
| **psycopg2-binary** | 2.9.9 | PostgreSQL driver |

---

## 11. External APIs & Integrations

### 11.1 Search & Data APIs
| Library | Version | Service |
|---------|---------|---------|
| **tavily-python** | 0.5.1 | Tavily search API |
| **duckduckgo-search** | >=7.2.0 | DuckDuckGo search |
| **wikipedia** | 1.4.0 | Wikipedia API |
| **arxiv** | 2.1.3 | ArXiv papers |
| **scholarly** | 1.7.11 | Google Scholar |
| **yfinance** | 0.2.65 | Yahoo Finance |
| **akshare** | >=1.15.78 | Chinese financial data |

### 11.2 Communication Platforms
| Library | Version | Platform |
|---------|---------|----------|
| **slack-sdk** | 3.37.0 | Slack |
| **discord-py** | 2.3.2 | Discord |
| **atlassian-python-api** | 4.0.7 | Jira/Confluence |

---

## 12. Observability & Debugging

| Library | Version | Mục đích |
|---------|---------|----------|
| **langfuse** | >=2.60.0 | LLM observability |
| **debugpy** | >=1.8.13 | Remote debugging |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Engine Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Document   │───>│    OCR &     │───>│   Chunking   │       │
│  │   Ingestion  │    │   Parsing    │    │  & Indexing  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                   │                    │                │
│        ▼                   ▼                    ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Tech Stack                             │   │
│  │  • pdfplumber, pypdf     • ONNX Runtime    • Elasticsearch│   │
│  │  • python-docx           • OpenCV          • Infinity     │   │
│  │  • Apache Tika           • HuggingFace     • Redis/Valkey │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Embedding  │───>│   Vector     │───>│   Retrieval  │       │
│  │   Generation │    │   Store      │    │   & Ranking  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                   │                    │                │
│        ▼                   ▼                    ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LLM Providers                          │   │
│  │  • OpenAI          • Anthropic        • Google           │   │
│  │  • Ollama          • ZhipuAI          • Cohere           │   │
│  │  • Mistral         • Dashscope        • VoyageAI         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Graph RAG  │    │   Response   │    │   Agent      │       │
│  │   (Optional) │───>│   Generation │───>│   Workflows  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Mục đích |
|------|----------|
| `rag/settings.py` | Configuration settings |
| `rag/llm/chat_model.py` | LLM chat abstraction |
| `rag/llm/embedding_model.py` | Embedding model abstraction |
| `rag/llm/rerank_model.py` | Reranking model abstraction |
| `rag/nlp/rag_tokenizer.py` | Custom tokenizer |
| `rag/utils/es_conn.py` | Elasticsearch connection |
| `rag/utils/redis_conn.py` | Redis connection |
| `rag/utils/minio_conn.py` | MinIO storage connection |
| `rag/utils/infinity_conn.py` | Infinity database connection |
| `deepdoc/vision/ocr.py` | OCR implementation |
| `deepdoc/parser/pdf_parser.py` | PDF parsing |
| `graphrag/general/graph_extractor.py` | Graph extraction |

---

## Summary

RAGFlow's RAG engine sử dụng một tech stack phong phú và đa dạng:

1. **Document Processing**: Sử dụng kết hợp pdfplumber, ONNX-based OCR, và nhiều document parsers
2. **NLP**: Custom tokenizer với hỗ trợ đa ngôn ngữ (đặc biệt English và Chinese)
3. **LLM**: Hỗ trợ 15+ LLM providers qua abstraction layer
4. **Vector Search**: Elasticsearch và Infinity với hybrid search capabilities
5. **Graph RAG**: NetworkX-based knowledge graph với LLM-powered extraction
6. **Async**: Trio-based structured concurrency cho parallel processing
7. **Storage**: Multi-cloud support với MinIO, S3, Azure Blob

Tech stack này cho phép RAGFlow xử lý documents ở quy mô lớn với khả năng tùy biến cao và hỗ trợ nhiều deployment scenarios.
