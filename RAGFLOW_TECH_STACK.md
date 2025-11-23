# RAGFlow - Technology Stack Documentation

Tài liệu chi tiết về các công nghệ được sử dụng trong RAGFlow, phân loại theo layers và chức năng.

---

## 📑 Mục Lục

1. [Architecture Layers Overview](#1-architecture-layers-overview)
2. [Frontend Layer](#2-frontend-layer)
3. [API Gateway Layer](#3-api-gateway-layer)
4. [Service Layer](#4-service-layer)
5. [RAG Engine Layer](#5-rag-engine-layer)
6. [Agentic System](#6-agentic-system)
7. [Data Layer](#7-data-layer)
8. [Infrastructure & DevOps](#8-infrastructure--devops)
9. [External Integrations](#9-external-integrations)
10. [Technology Decision Matrix](#10-technology-decision-matrix)

---

## 1. Architecture Layers Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                            │
│  React • TypeScript • Ant Design • Zustand • TailwindCSS    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/SSE
┌────────────────────────▼────────────────────────────────────┐
│                  API GATEWAY LAYER                           │
│  Quart (Async Flask) • CORS • JWT • Flasgger (Swagger)      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼──────┐  ┌──────▼────────┐  ┌──▼────────────┐
│  SERVICE     │  │  RAG ENGINE   │  │  AGENTIC      │
│  LAYER       │  │  LAYER        │  │  SYSTEM       │
│              │  │               │  │               │
│ Business     │  │ Embedding     │  │ Canvas        │
│ Logic        │  │ Retrieval     │  │ Components    │
│ Validation   │  │ Reranking     │  │ Tools         │
│ Orchestration│  │ Generation    │  │ Workflow      │
└───────┬──────┘  └──────┬────────┘  └──┬────────────┘
        │                │              │
        └────────────────┼──────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      DATA LAYER                              │
│  MySQL/PostgreSQL • Elasticsearch/Infinity • Redis • MinIO   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend Layer

### 2.1. Core Framework

| Technology | Version | Purpose | Why Chosen |
|------------|---------|---------|------------|
| **React** | 18.x | UI framework | Component-based, large ecosystem, performance |
| **TypeScript** | 5.x | Type safety | Better IDE support, catch errors early |
| **UmiJS** | 4.x | React framework | Convention over configuration, plugins |

### 2.2. UI Components

| Technology | Purpose | Key Features |
|------------|---------|--------------|
| **Ant Design** | Component library | Enterprise-grade UI, comprehensive components |
| **shadcn/ui** | Modern components | Customizable, accessible, Radix UI based |
| **Tailwind CSS** | Utility-first CSS | Rapid styling, consistent design |
| **React Hook Form** | Form management | Performance, validation, less re-renders |

### 2.3. State Management

| Technology | Purpose | Use Case |
|------------|---------|----------|
| **Zustand** | Global state | Simple API, no boilerplate, TypeScript support |
| **React Query** | Server state | Caching, auto-refresh, optimistic updates |
| **Context API** | Local state | Theme, auth, simple shared state |

### 2.4. Build & Dev Tools

```javascript
{
  "dependencies": {
    "react": "^18.2.0",
    "typescript": "^5.0.0",
    "@umijs/max": "^4.0.0",
    "antd": "^5.12.0",
    "zustand": "^4.4.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",        // Fast build tool
    "eslint": "^8.55.0",     // Linting
    "prettier": "^3.1.0",    // Code formatting
    "jest": "^29.7.0",       // Testing
    "@testing-library/react": "^14.0.0"
  }
}
```

**Build Pipeline:**
- **Vite** - Lightning-fast HMR, ES modules
- **PostCSS** - CSS transformations
- **Babel** - JavaScript transpilation (via UmiJS)

---

## 3. API Gateway Layer

### 3.1. Web Framework

| Technology | Version | Purpose | Key Features |
|------------|---------|---------|--------------|
| **Quart** | 0.19.x | Async web framework | Async/await, Flask-compatible API, HTTP/2 support |
| **Werkzeug** | 3.0.x | WSGI utilities | Request/response handling, routing |

**Why Quart over Flask?**
- ✅ Native async/await support
- ✅ Better performance for I/O-bound operations
- ✅ Compatible with Flask ecosystem
- ✅ SSE (Server-Sent Events) support

### 3.2. API Documentation

| Technology | Purpose |
|------------|---------|
| **Flasgger** | Swagger/OpenAPI UI |
| **Swagger 2.0** | API specification |
| **JSON Schema** | Request/response validation |

**Swagger UI URL:** `http://localhost:9380/apidocs/`

### 3.3. Authentication & Security

| Technology | Purpose | Implementation |
|------------|---------|----------------|
| **itsdangerous** | Token signing | JWT token generation/verification |
| **CORS** | Cross-origin requests | `quart-cors` middleware |
| **Rate Limiting** | API protection | Redis-based rate limiter |
| **Input Validation** | Security | Custom validators + JSON Schema |

**Authentication Flow:**
```python
# JWT Token Structure
{
    "user_id": "uuid",
    "tenant_id": "uuid",
    "exp": 1234567890,  # Expiration
    "iat": 1234567890   # Issued at
}

# API Token Structure
{
    "token": "ragflow-xxxxxxxx",
    "tenant_id": "uuid",
    "created_at": timestamp
}
```

### 3.4. Session Management

| Technology | Purpose |
|------------|---------|
| **Redis** | Session storage |
| **Flask-Session** | Session interface |
| **Cookies** | Client-side session ID |

---

## 4. Service Layer

### 4.1. ORM & Database Access

| Technology | Version | Purpose | Features |
|------------|---------|---------|----------|
| **Peewee** | 3.17.x | ORM | Lightweight, expressive, Python-native |
| **PyMySQL** | 1.1.x | MySQL driver | Pure Python, no C dependencies |
| **psycopg2** | 2.9.x | PostgreSQL driver | Fast, mature |

**Why Peewee?**
- ✅ Simpler than SQLAlchemy
- ✅ Better for microservices
- ✅ Excellent query builder
- ✅ Connection pooling support

**Model Example:**
```python
from peewee import *

class Document(Model):
    id = CharField(primary_key=True, max_length=32)
    kb_id = CharField(max_length=32, index=True)
    name = CharField(max_length=255)
    parser_id = CharField(max_length=32)
    parser_config = TextField()
    chunk_num = IntegerField(default=0)
    token_num = BigIntegerField(default=0)
    create_time = BigIntegerField()

    class Meta:
        database = DB
        table_name = 'document'
```

### 4.2. Background Task Processing

| Technology | Purpose | Features |
|------------|---------|----------|
| **Trio** | Async concurrency | Structured concurrency, easy to reason about |
| **Redis Queue** | Task queue | Reliable, persistent, consumer groups |
| **Celery** | (Alternative) | Distributed task queue (not used currently) |

**Concurrency Model:**
```python
import trio

# Concurrency limits
task_limiter = trio.Semaphore(5)        # Max 5 tasks
chunk_limiter = trio.CapacityLimiter(1) # Max 1 chunker
embed_limiter = trio.CapacityLimiter(1) # Max 1 embedder

async def process_task(task):
    async with task_limiter:
        # Process document
        await parse_document()
        async with chunk_limiter:
            await chunk_document()
        async with embed_limiter:
            await embed_chunks()
```

### 4.3. Service Architecture Pattern

```python
# Base Service Class
class CommonService:
    model = None  # Peewee model

    @classmethod
    @DB.connection_context()
    def get_by_id(cls, id):
        try:
            obj = cls.model.get_by_id(id)
            return True, obj
        except cls.model.DoesNotExist:
            return False, None

    @classmethod
    def query(cls, **kwargs):
        return cls.model.select().where(
            *[getattr(cls.model, k) == v for k, v in kwargs.items()]
        )

# Concrete Service
class DocumentService(CommonService):
    model = Document

    @classmethod
    def update_progress(cls):
        # Custom business logic
        pass
```

---

## 5. RAG Engine Layer

### 5.1. Document Processing

#### 5.1.1. Text Extraction

| Technology | Purpose | Supported Formats |
|------------|---------|-------------------|
| **deepdoc** | PDF parsing | PDF, scanned PDFs |
| **pdfplumber** | PDF text extraction | PDF |
| **python-docx** | Word documents | DOCX, DOC |
| **openpyxl** | Excel files | XLSX, XLS |
| **python-pptx** | PowerPoint | PPTX, PPT |
| **BeautifulSoup** | HTML parsing | HTML, XML |
| **Markdown** | Markdown parsing | MD |

#### 5.1.2. OCR & Vision

| Technology | Purpose | Languages |
|------------|---------|-----------|
| **PaddleOCR** | OCR engine | 80+ languages, Chinese-optimized |
| **Tesseract** | OCR alternative | 100+ languages |
| **PIL/Pillow** | Image processing | Format conversion, resizing |
| **OpenCV** | Computer vision | Layout detection, preprocessing |

#### 5.1.3. Audio Processing

| Technology | Purpose | Models |
|------------|---------|--------|
| **Whisper** | Speech-to-text | tiny, base, small, medium, large |
| **faster-whisper** | Optimized Whisper | CTranslate2-based, faster inference |
| **pydub** | Audio manipulation | Format conversion, trimming |

### 5.2. NLP & Text Processing

| Technology | Purpose | Features |
|------------|---------|----------|
| **NLTK** | NLP toolkit | Tokenization, stemming, stopwords |
| **jieba** | Chinese segmentation | Accurate, fast, custom dictionary |
| **spaCy** | Advanced NLP | NER, POS tagging, dependency parsing |
| **langdetect** | Language detection | 55+ languages |

**Tokenization Pipeline:**
```python
# English
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

tokens = word_tokenize(text)
stemmer = PorterStemmer()
stems = [stemmer.stem(t) for t in tokens]

# Chinese
import jieba
tokens = jieba.cut(text)
```

### 5.3. Embedding Models

| Model Family | Dimensions | Use Case | Performance |
|--------------|-----------|----------|-------------|
| **BGE (BAAI)** | 768/1024 | Chinese/English, SOTA | Best |
| **BERT** | 768 | General purpose | Good |
| **Sentence-BERT** | 384/768 | Sentence similarity | Fast |
| **OpenAI Ada** | 1536 | General purpose, API | Good |
| **Jina Embeddings** | 768 | Multilingual | Very Good |
| **M3E** | 768 | Chinese-focused | Best (Chinese) |

**Model Loading:**
```python
from sentence_transformers import SentenceTransformer

# Local model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings = model.encode(texts)

# API-based
from openai import OpenAI
client = OpenAI()
response = client.embeddings.create(
    input=texts,
    model="text-embedding-ada-002"
)
```

### 5.4. LLM Integration

#### 5.4.1. LiteLLM - Universal LLM Gateway

| Feature | Description |
|---------|-------------|
| **Providers** | 30+ LLM providers unified API |
| **Models** | OpenAI, Anthropic, Azure, Cohere, Replicate, etc. |
| **Features** | Retry logic, fallbacks, load balancing, caching |

**Supported Providers:**
```python
SUPPORTED_PROVIDERS = {
    # Commercial APIs
    "openai": ["gpt-4", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet"],
    "azure": ["gpt-4", "gpt-35-turbo"],
    "cohere": ["command", "command-light"],
    "replicate": ["llama-2-70b", "mistral-7b"],

    # Open Source (Self-hosted)
    "ollama": ["llama2", "mistral", "mixtral"],
    "xinference": ["chatglm3", "qwen"],
    "vllm": ["mistral-7b", "llama-2-13b"],

    # Chinese Providers
    "zhipu": ["chatglm-turbo", "chatglm-pro"],
    "minimax": ["abab5.5-chat"],
    "baichuan": ["baichuan2-53b"],
    "moonshot": ["moonshot-v1-8k"]
}
```

#### 5.4.2. Model Types

| Type | Purpose | Example Models |
|------|---------|----------------|
| **Chat** | Conversation | GPT-4, Claude-3, LLaMA-2 |
| **Embedding** | Vector generation | BGE, Ada-002, M3E |
| **Rerank** | Result reordering | BGE-reranker, Cohere-rerank |
| **Image2Text** | Vision | GPT-4V, Claude-3, LLaVA |
| **TTS** | Text-to-Speech | OpenAI TTS, Azure TTS |
| **ASR** | Speech-to-Text | Whisper, Azure STT |

#### 5.4.3. LLM Configuration

```python
# api/db/services/llm_service.py
class LLMBundle:
    def __init__(self, tenant_id, llm_type, llm_id):
        self.tenant_id = tenant_id
        self.llm_type = llm_type
        self.llm_id = llm_id

        # Load API key and config
        self.api_key = TenantLLMService.get_api_key(tenant_id, llm_id)
        self.base_url = TenantLLMService.get_base_url(tenant_id, llm_id)
        self.model_name = TenantLLMService.get_model_name(tenant_id, llm_id)

    def chat(self, system, messages, gen_conf):
        # LiteLLM unified interface
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                *messages
            ],
            api_key=self.api_key,
            api_base=self.base_url,
            **gen_conf
        )
        return response.choices[0].message.content
```

### 5.5. Vector Search

#### 5.5.1. Elasticsearch

| Feature | Configuration | Purpose |
|---------|--------------|---------|
| **Version** | 8.x | Vector search support |
| **Vector Type** | dense_vector | Store embeddings |
| **Similarity** | cosine, dot_product | Distance metric |
| **Index** | HNSW | Fast ANN search |

**Index Mapping:**
```json
{
  "mappings": {
    "properties": {
      "q_vec": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      },
      "content_tks": {
        "type": "text",
        "analyzer": "ik_max_word"
      },
      "title_tks": {
        "type": "text",
        "analyzer": "ik_max_word",
        "boost": 8
      }
    }
  }
}
```

#### 5.5.2. Infinity (Alternative)

| Feature | Description |
|---------|-------------|
| **Purpose** | Lightweight vector DB |
| **Performance** | Faster than ES for pure vector search |
| **Storage** | Embedded database, no separate service |
| **API** | PostgreSQL-compatible |

**When to use Infinity:**
- ✅ Smaller deployments (< 10M vectors)
- ✅ Want embedded solution
- ✅ Pure vector search (no full-text)
- ❌ Need advanced full-text features

### 5.6. Reranking

| Model | Provider | Use Case |
|-------|----------|----------|
| **bge-reranker-large** | BAAI | Chinese/English, best accuracy |
| **bge-reranker-base** | BAAI | Faster, good accuracy |
| **Cohere Rerank** | Cohere API | Multilingual, easy to use |
| **Jina Reranker** | Jina API | Fast, good for English |

**Reranking Pipeline:**
```python
def rerank(query, chunks, top_k=6):
    # Get relevance scores
    pairs = [(query, chunk['content']) for chunk in chunks]
    scores = rerank_model.predict(pairs)

    # Sort by score
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top_k
    return [chunk for chunk, score in ranked[:top_k]]
```

### 5.7. Prompt Engineering

| Technology | Purpose |
|------------|---------|
| **Jinja2** | Template engine |
| **LangChain (partial)** | Prompt templates (not fully integrated) |
| **Custom templates** | Domain-specific prompts |

**Template Example:**
```jinja2
{# rag/prompts/rag.jinja2 #}
You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question.

{% if knowledge %}
Here is the knowledge base:
{% for chunk in knowledge %}
[ID:{{ loop.index0 }}] {{ chunk.content }}
{% endfor %}
{% endif %}

Please cite sources using [ID:N] format in your answer.

{% if language != "English" %}
Please answer in {{ language }}.
{% endif %}

Question: {{ question }}
```

---

## 6. Agentic System

### 6.1. Core Framework

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Workflow Engine** | Custom (Canvas) | DAG execution |
| **Component System** | Plugin-based | Modular components |
| **Tool Registry** | Dynamic loading | External tool integration |
| **State Management** | Redis | Distributed state |

### 6.2. Agentic Components

#### 6.2.1. Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Begin** | Workflow entry | Input validation, initialization |
| **LLM** | Language model invocation | Streaming, prompt templates, variable substitution |
| **Retrieval** | Knowledge base search | RAG integration, multi-KB support |
| **Categorize** | Intent classification | Dynamic routing, LLM-based |
| **Switch** | Conditional routing | Expression evaluation, multi-path |
| **Iteration** | Loop execution | Max iterations, early exit |
| **Variable Assigner** | State mutation | Set variables, data transformation |
| **Answer** | Workflow output | Format response, cleanup |
| **Message** | User interaction | Rich content, buttons, forms |
| **Invoke** | Sub-workflow call | Nested workflows, reusability |

#### 6.2.2. Component Architecture

```python
# agent/component/base.py
class ComponentBase:
    def __init__(self, canvas, component_id, component_obj):
        self.canvas = canvas
        self.id = component_id
        self.params = component_obj.get("params", {})
        self.downstream = component_obj.get("downstream", [])
        self.upstream = component_obj.get("upstream", [])

    async def invoke(self, **kwargs):
        """
        Execute component logic

        Args:
            **kwargs: Inputs from upstream

        Returns:
            dict: {
                "output": component_output,
                "downstream": [next_component_ids]
            }
        """
        raise NotImplementedError

    def callback(self, message, progress=0.0):
        """Report progress to user"""
        self.canvas.on_progress(self.id, message, progress)
```

**Example: LLM Component**
```python
# agent/component/llm.py
class LLM(ComponentBase):
    async def invoke(self, **kwargs):
        # Render prompt with variables
        prompt = self.render_prompt(
            self.params["prompt"],
            kwargs
        )

        # Get LLM instance
        llm = LLMBundle(
            tenant_id=self.canvas.tenant_id,
            llm_type=LLMType.CHAT,
            llm_id=self.params["llm_id"]
        )

        # Generate
        if self.params.get("stream"):
            answer = ""
            for chunk in llm.chat_streamly(prompt, self.params):
                answer += chunk
                self.callback(chunk, progress)
            return {"output": answer}
        else:
            answer = llm.chat(prompt, self.params)
            return {"output": answer}
```

### 6.3. Tool Integrations

#### 6.3.1. Built-in Tools

| Tool | Technology | Purpose |
|------|------------|---------|
| **retrieval** | RAG Engine | Search knowledge bases |
| **tavily** | Tavily API | Web search |
| **wikipedia** | wikipedia-api | Wikipedia search |
| **arxiv** | arxiv API | Academic papers |
| **pubmed** | Bio.Entrez | Medical literature |
| **google** | Custom Search API | Google search |
| **duckduckgo** | duckduckgo_search | Privacy-focused search |
| **code_exec** | subprocess | Code execution sandbox |
| **exesql** | SQLAlchemy | SQL database queries |
| **email** | SMTP | Send emails |
| **crawler** | Selenium/BeautifulSoup | Web scraping |

#### 6.3.2. Tool Implementation Pattern

```python
# agent/tools/base.py
class ToolBase:
    def __init__(self, **config):
        self.config = config

    def validate_params(self, params):
        """Validate input parameters"""
        raise NotImplementedError

    def execute(self, **params):
        """Execute tool logic"""
        raise NotImplementedError

# agent/tools/tavily.py
class TavilyTool(ToolBase):
    def __init__(self, api_key):
        self.api_key = api_key

    def execute(self, query, max_results=5):
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced"
            }
        )
        return {
            "results": response.json()["results"],
            "query": query
        }
```

#### 6.3.3. Tool Registry

```python
# agent/tools/manager.py
class ToolManager:
    def __init__(self):
        self.tools = {}

    def register(self, name, tool_class, **config):
        self.tools[name] = tool_class(**config)

    def execute(self, tool_name, **params):
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        tool = self.tools[tool_name]
        tool.validate_params(params)
        return tool.execute(**params)

# Usage
tool_manager = ToolManager()
tool_manager.register("tavily", TavilyTool, api_key=settings.TAVILY_API_KEY)
tool_manager.register("retrieval", RetrievalTool, dialog_service=DialogService)

result = tool_manager.execute("tavily", query="What is RAG?")
```

### 6.4. Workflow Execution

#### 6.4.1. Canvas DSL

```python
# Workflow Definition
{
    "components": {
        "begin": {
            "obj": {
                "component_name": "Begin",
                "params": {}
            },
            "downstream": ["categorize_0"],
            "upstream": []
        },
        "categorize_0": {
            "obj": {
                "component_name": "Categorize",
                "params": {
                    "categories": ["technical", "general"],
                    "llm_id": "gpt-4"
                }
            },
            "downstream": ["retrieval_0", "llm_0"],
            "upstream": ["begin"]
        },
        "retrieval_0": {
            "obj": {
                "component_name": "Retrieval",
                "params": {
                    "kb_ids": ["kb_123"],
                    "top_k": 5
                }
            },
            "downstream": ["answer"],
            "upstream": ["categorize_0"]
        },
        "llm_0": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "llm_id": "gpt-3.5-turbo",
                    "prompt": "Answer directly: {query}"
                }
            },
            "downstream": ["answer"],
            "upstream": ["categorize_0"]
        },
        "answer": {
            "obj": {
                "component_name": "Answer",
                "params": {}
            },
            "downstream": [],
            "upstream": ["retrieval_0", "llm_0"]
        }
    },
    "globals": {
        "sys.query": "",
        "sys.user_id": "",
        "sys.conversation_turns": 0
    }
}
```

#### 6.4.2. Execution Engine

```python
# agent/canvas.py
class Canvas:
    def __init__(self, dsl, tenant_id):
        self.dsl = dsl
        self.tenant_id = tenant_id
        self.components = {}
        self.graph = None

    async def run(self, **kwargs):
        # 1. Build execution graph
        self.build_graph()

        # 2. Initialize globals
        self.globals = self.dsl.get("globals", {})
        self.globals.update(kwargs)

        # 3. Execute from Begin
        current = "begin"
        outputs = self.globals.copy()

        while current:
            # Get component
            component = self.get_component(current)

            # Execute
            result = await component.invoke(**outputs)

            # Merge outputs
            outputs.update(result.get("output", {}))

            # Get next component
            downstream = result.get("downstream", [])
            current = downstream[0] if downstream else None

        return outputs

    def build_graph(self):
        # Build DAG from DSL
        # Validate no cycles
        # Topological sort
        pass
```

### 6.5. Agentic Patterns

#### 6.5.1. ReAct (Reasoning + Acting)

```python
# Implemented via Loop + LLM + Tools
{
    "components": {
        "begin": {...},
        "iteration_0": {
            "obj": {
                "component_name": "Iteration",
                "params": {
                    "max_iterations": 5,
                    "loop_body": ["think", "act", "observe"]
                }
            },
            "downstream": ["think"]
        },
        "think": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "prompt": "Think about next action: {observation}"
                }
            },
            "downstream": ["act"]
        },
        "act": {
            "obj": {
                "component_name": "ToolCall",
                "params": {
                    "tool_selection": "from_llm_output"
                }
            },
            "downstream": ["observe"]
        },
        "observe": {
            "obj": {
                "component_name": "VariableAssigner",
                "params": {
                    "observation": "{tool_output}"
                }
            },
            "downstream": ["iteration_0"]  # Loop back
        }
    }
}
```

#### 6.5.2. Chain-of-Thought (CoT)

```python
# Implemented via Sequential LLM calls
{
    "components": {
        "begin": {...},
        "step1_decompose": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "prompt": "Break down the problem: {query}"
                }
            }
        },
        "step2_solve_each": {
            "obj": {
                "component_name": "Iteration",
                "params": {
                    "iterate_over": "{sub_problems}",
                    "loop_body": ["solve_subproblem"]
                }
            }
        },
        "step3_synthesize": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "prompt": "Combine solutions: {sub_solutions}"
                }
            }
        }
    }
}
```

#### 6.5.3. Self-Consistency

```python
# Multiple reasoning paths + voting
{
    "components": {
        "begin": {...},
        "generate_paths": {
            "obj": {
                "component_name": "Iteration",
                "params": {
                    "max_iterations": 5,
                    "parallel": true
                }
            },
            "downstream": ["llm_reason"]
        },
        "llm_reason": {
            "obj": {
                "component_name": "LLM",
                "params": {
                    "temperature": 0.7,  # More diverse
                    "prompt": "Solve: {query}"
                }
            },
            "downstream": ["vote"]
        },
        "vote": {
            "obj": {
                "component_name": "MajorityVote",
                "params": {
                    "answers": "{reasoning_paths}"
                }
            }
        }
    }
}
```

### 6.6. Advanced Agentic Features

#### 6.6.1. DeepResearch Integration

```python
# agentic_reasoning library
from agentic_reasoning import DeepResearcher

researcher = DeepResearcher(
    llm=chat_model,
    search_engine=tavily_tool,
    max_depth=3,
    max_iterations=10
)

result = researcher.research(
    query="Compare RAG architectures",
    knowledge_bases=["kb_123"],
    output_format="report"
)
```

**Features:**
- ✅ Multi-step reasoning
- ✅ Web + KB search integration
- ✅ Iterative refinement
- ✅ Citation tracking
- ✅ Report generation

#### 6.6.2. Graph RAG

```python
# graphrag/ module
from graphrag.general.mind_map_extractor import MindMapExtractor

extractor = MindMapExtractor(
    llm=chat_model,
    knowledge_graph=neo4j_conn
)

mindmap = extractor.extract(
    query="What is RAG?",
    kb_ids=["kb_123"],
    max_depth=2
)
```

**Features:**
- ✅ Entity extraction
- ✅ Relationship mapping
- ✅ Graph traversal
- ✅ Mind map generation

---

## 7. Data Layer

### 7.1. Relational Database

| Technology | Version | Purpose | Features |
|------------|---------|---------|----------|
| **MySQL** | 8.0+ | Primary database | ACID, transactions, indexes |
| **PostgreSQL** | 14+ | Alternative | Advanced features, JSON support |

**Database Configuration:**
```yaml
# docker/service_conf.yaml
mysql:
  name: rag_flow
  user: ragflow
  password: ragflow_password
  host: mysql
  port: 3306
  max_connections: 100
  pool_size: 10
```

**Connection Pooling:**
```python
from peewee import PooledMySQLDatabase

DB = PooledMySQLDatabase(
    database=settings.MYSQL_DATABASE,
    user=settings.MYSQL_USER,
    password=settings.MYSQL_PASSWORD,
    host=settings.MYSQL_HOST,
    port=settings.MYSQL_PORT,
    max_connections=100,
    stale_timeout=300
)
```

### 7.2. Vector Database

#### 7.2.1. Elasticsearch

**Version:** 8.11+

**Key Features:**
- ✅ Vector search (dense_vector)
- ✅ Full-text search (BM25)
- ✅ Hybrid search
- ✅ Aggregations
- ✅ Filtering

**Configuration:**
```yaml
elasticsearch:
  hosts:
    - http://es01:9200
  user: elastic
  password: elastic_password
  scheme: http
  vector_similarity: cosine
  number_of_shards: 1
  number_of_replicas: 0
```

#### 7.2.2. Infinity (Lightweight Alternative)

**Version:** Latest

**Key Features:**
- ✅ Embedded vector DB
- ✅ PostgreSQL-compatible
- ✅ Fast ANN search
- ✅ No separate service needed

**When to use:**
```
Use Elasticsearch when:
  - Need advanced full-text
  - Need aggregations
  - Large scale (10M+ vectors)

Use Infinity when:
  - Small/medium scale
  - Want simplicity
  - Pure vector search
  - Embedded deployment
```

### 7.3. Cache & Queue

#### 7.3.1. Redis

**Version:** 7.x

**Use Cases:**
```python
# 1. Session storage
app.config["SESSION_REDIS"] = redis_client

# 2. Task queue
redis_client.lpush("task_queue", task_json)

# 3. Distributed lock
from rag.utils.redis_conn import RedisDistributedLock
lock = RedisDistributedLock("resource_key", timeout=60)

# 4. Cache
@cache.memoize(timeout=300)
def expensive_operation():
    pass

# 5. Rate limiting
redis_client.incr(f"rate_limit:{user_id}:{minute}")

# 6. Progress tracking
redis_client.set(f"progress:{task_id}", json.dumps(progress))

# 7. Canvas execution trace
redis_client.lpush(f"canvas:{canvas_id}:trace", event_json)
```

**Configuration:**
```yaml
redis:
  host: redis
  port: 6379
  db: 0
  password: ""
  max_connections: 50
```

### 7.4. Object Storage

| Technology | Purpose | Protocol |
|------------|---------|----------|
| **MinIO** | Self-hosted S3 | S3 API |
| **AWS S3** | Cloud storage | S3 API |
| **Azure Blob** | Cloud storage | Azure API |
| **Local FileSystem** | Development | File I/O |

**Storage Interface:**
```python
# common/file_utils.py
class StorageBackend:
    def put(self, bucket, key, data):
        raise NotImplementedError

    def get(self, bucket, key):
        raise NotImplementedError

    def delete(self, bucket, key):
        raise NotImplementedError

    def obj_exist(self, bucket, key):
        raise NotImplementedError

# MinIO Implementation
class MinIOStorage(StorageBackend):
    def __init__(self, endpoint, access_key, secret_key):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key
        )

    def put(self, bucket, key, data):
        self.client.put_object(
            bucket_name=bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data)
        )
```

**Usage:**
```python
# Store document
settings.STORAGE_IMPL.put(
    bucket=kb_id,
    key=doc_location,
    data=file_blob
)

# Retrieve document
blob = settings.STORAGE_IMPL.get(
    bucket=kb_id,
    key=doc_location
)
```

---

## 8. Infrastructure & DevOps

### 8.1. Containerization

| Technology | Purpose |
|------------|---------|
| **Docker** | Container runtime |
| **Docker Compose** | Multi-container orchestration |
| **Docker BuildKit** | Optimized builds |

**Docker Architecture:**
```yaml
# docker/docker-compose.yml
services:
  ragflow-server:
    image: infiniflow/ragflow:nightly
    container_name: ragflow-server
    environment:
      - MYSQL_HOST=mysql
      - ES_HOSTS=http://es01:9200
      - REDIS_HOST=redis
      - MINIO_HOST=minio:9000
    depends_on:
      - mysql
      - es01
      - redis
      - minio

  mysql:
    image: mysql:8.0
    container_name: ragflow-mysql
    volumes:
      - mysql_data:/var/lib/mysql

  es01:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.3
    container_name: ragflow-es
    environment:
      - xpack.security.enabled=false
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data

  redis:
    image: redis:7-alpine
    container_name: ragflow-redis
    volumes:
      - redis_data:/data

  minio:
    image: minio/minio:latest
    container_name: ragflow-minio
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  mysql_data:
  es_data:
  redis_data:
  minio_data:
```

### 8.2. Package Management

| Technology | Purpose |
|------------|---------|
| **uv** | Fast Python package installer (Rust-based) |
| **pip** | Traditional Python packages |
| **npm** | Frontend packages |

**pyproject.toml:**
```toml
[project]
name = "ragflow"
version = "0.9.0"
requires-python = ">=3.10,<3.13"

dependencies = [
    "quart>=0.19.0",
    "peewee>=3.17.0",
    "elasticsearch>=8.11.0",
    "redis>=5.0.0",
    "litellm>=1.30.0",
    "sentence-transformers>=2.2.0",
    "trio>=0.24.0",
    "deepdoc>=0.1.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0"
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "ruff>=0.1.0"
]
```

**Installation:**
```bash
# Using uv (fast)
uv sync --python 3.10 --all-extras

# Traditional pip
pip install -e ".[dev]"
```

### 8.3. Testing

| Technology | Purpose |
|------------|---------|
| **pytest** | Test framework |
| **pytest-asyncio** | Async test support |
| **pytest-cov** | Coverage reporting |
| **unittest.mock** | Mocking |

**Test Structure:**
```
test/
├── test_api/
│   ├── test_document_api.py
│   ├── test_conversation_api.py
│   └── test_canvas_api.py
├── test_services/
│   ├── test_dialog_service.py
│   └── test_document_service.py
├── test_rag/
│   ├── test_embedding.py
│   ├── test_search.py
│   └── test_rerank.py
└── conftest.py
```

**Example Test:**
```python
import pytest
from api.db.services.document_service import DocumentService

@pytest.mark.asyncio
async def test_document_upload(client, test_kb):
    # Upload document
    response = await client.post(
        "/v1/document/upload",
        data={
            "kb_id": test_kb["id"],
            "file": (io.BytesIO(b"test content"), "test.txt")
        }
    )

    assert response.status_code == 200
    data = await response.get_json()
    assert len(data["data"]) == 1

    # Verify in database
    doc = DocumentService.get_by_id(data["data"][0]["id"])
    assert doc.name == "test.txt"
```

### 8.4. Code Quality

| Technology | Purpose |
|------------|---------|
| **Ruff** | Fast linter (Rust-based) |
| **Black** | Code formatter |
| **mypy** | Type checking |
| **pre-commit** | Git hooks |

**Configuration:**
```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501", "N806"]

[tool.black]
line-length = 120
target-version = ["py310"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

### 8.5. Monitoring & Logging

| Technology | Purpose |
|------------|---------|
| **Logging** | Built-in Python logging |
| **Langfuse** | LLM observability |
| **Prometheus** | Metrics (future) |
| **Grafana** | Dashboards (future) |

**Logging Configuration:**
```python
# common/log_utils.py
import logging
from logging.handlers import RotatingFileHandler

def init_root_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # File handler
    file_handler = RotatingFileHandler(
        f"logs/{name}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
```

**Langfuse Integration:**
```python
# api/db/services/langfuse_service.py
from langfuse import Langfuse

class TenantLangfuseService:
    @staticmethod
    def get_client(tenant_id):
        config = TenantLangfuseService.get_config(tenant_id)
        if not config:
            return None

        return Langfuse(
            public_key=config["public_key"],
            secret_key=config["secret_key"],
            host=config.get("host", "https://cloud.langfuse.com")
        )

# Usage in chat
langfuse = TenantLangfuseService.get_client(tenant_id)
trace = langfuse.trace(name="chat")
span = trace.span(name="retrieval")
span.end(output=chunks)
```

---

## 9. External Integrations

### 9.1. LLM Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | GPT-4, GPT-3.5 | General purpose, high quality |
| **Anthropic** | Claude-3 | Long context, safety |
| **Azure OpenAI** | GPT-4, Ada | Enterprise |
| **Ollama** | LLaMA, Mistral | Self-hosted, privacy |
| **Xinference** | Chinese models | Chinese language |
| **Cohere** | Command | Multilingual |
| **Replicate** | Open source models | Flexible |

### 9.2. Embedding Providers

| Provider | Models | Dimensions |
|----------|--------|-----------|
| **OpenAI** | text-embedding-ada-002 | 1536 |
| **Jina AI** | jina-embeddings-v2 | 768 |
| **Cohere** | embed-multilingual-v3.0 | 1024 |
| **HuggingFace** | BGE, M3E, etc. | 768/1024 |
| **Local** | Sentence-BERT | 384/768 |

### 9.3. Search APIs

| Service | Purpose | API |
|---------|---------|-----|
| **Tavily** | Web search | REST API |
| **Google Custom Search** | Google search | REST API |
| **DuckDuckGo** | Privacy search | Python library |
| **Wikipedia** | Encyclopedia | wikipedia-api |
| **arXiv** | Academic papers | arxiv API |
| **PubMed** | Medical literature | Bio.Entrez |

### 9.4. Data Source Connectors

**Supported Connectors** (from `api/apps/connector_app.py`):
- **Database:** MySQL, PostgreSQL, SQL Server, Oracle
- **Cloud Storage:** S3, Azure Blob, Google Cloud Storage
- **SaaS:** Notion, Confluence, SharePoint, Google Drive
- **Messaging:** Slack, Discord, Teams
- **Email:** Gmail, Outlook, IMAP
- **CRM:** Salesforce, HubSpot
- **Project Management:** Jira, Asana, Trello

---

## 10. Technology Decision Matrix

### 10.1. Key Technology Choices

| Decision | Options Considered | Chosen | Reason |
|----------|-------------------|--------|--------|
| **Web Framework** | Flask, FastAPI, Quart | **Quart** | Async + Flask compatibility |
| **ORM** | SQLAlchemy, Peewee, Django ORM | **Peewee** | Lightweight, simpler API |
| **Vector DB** | Pinecone, Weaviate, ES, Infinity | **Elasticsearch + Infinity** | Full-text + vector, flexibility |
| **LLM Gateway** | LangChain, LlamaIndex, LiteLLM | **LiteLLM** | Simple, multi-provider, no lock-in |
| **Task Queue** | Celery, RQ, Trio | **Trio + Redis** | Structured concurrency, simpler |
| **Frontend** | Vue, React, Svelte | **React** | Ecosystem, team expertise |
| **State Mgmt** | Redux, MobX, Zustand | **Zustand** | Simplicity, no boilerplate |
| **Embedding** | OpenAI, BGE, Sentence-BERT | **BGE** | SOTA for Chinese, open source |

### 10.2. Scalability Considerations

| Component | Current | Scalability Path |
|-----------|---------|------------------|
| **API Server** | Single instance | → Load balancer + multiple instances |
| **Task Workers** | Single instance | → Worker pool with autoscaling |
| **MySQL** | Single instance | → Read replicas, sharding |
| **Elasticsearch** | Single node | → Multi-node cluster |
| **Redis** | Single instance | → Redis Cluster, Sentinel |
| **Storage** | MinIO single | → MinIO distributed, S3 |

### 10.3. Performance Characteristics

| Operation | Latency | Throughput | Bottleneck |
|-----------|---------|------------|------------|
| **Document Upload** | < 1s | 10/s | Storage I/O |
| **Document Parsing** | 5-30s | 5 concurrent | CPU (OCR) |
| **Embedding Generation** | 1-5s/batch | 100 texts/s | GPU/API |
| **Vector Search** | < 100ms | 1000 QPS | Elasticsearch |
| **LLM Chat** | 2-10s | 100 RPS | LLM API |
| **Reranking** | 200-500ms | 50 RPS | Rerank model |

---

## 📚 Technology Stack Summary

```
┌───────────────── FRONTEND ─────────────────┐
│ React 18 • TypeScript • UmiJS • Ant Design │
│ Zustand • Tailwind CSS • Vite              │
└────────────────────┬───────────────────────┘
                     │
┌────────────────── API ─────────────────────┐
│ Quart • Flasgger • JWT • CORS              │
└────────────────────┬───────────────────────┘
                     │
┌─────────────── SERVICE ────────────────────┐
│ Peewee ORM • Trio • Redis Queue            │
└────────────────────┬───────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──┐  ┌──────▼───┐  ┌────▼──────┐
│   RAG    │  │ AGENTIC  │  │   DATA    │
│          │  │          │  │           │
│ LiteLLM  │  │ Canvas   │  │ MySQL     │
│ BGE      │  │ Tools    │  │ ES        │
│ ES/Inf.  │  │ DeepRes. │  │ Redis     │
│ deepdoc  │  │ GraphRAG │  │ MinIO     │
└──────────┘  └──────────┘  └───────────┘
```

---

**Generated:** 2025-11-23
**Version:** RAGFlow 0.9.0
