# RAGFlow - PlantUML Diagrams

Tài liệu này chứa các PlantUML diagrams chi tiết cho RAGFlow system.

## Mục Lục
1. [Architecture Overview](#1-architecture-overview)
2. [Document Processing Flow](#2-document-processing-flow)
3. [RAG Retrieval Flow](#3-rag-retrieval-flow)
4. [Chat Conversation Flow](#4-chat-conversation-flow)
5. [Agent Workflow Execution](#5-agent-workflow-execution)
6. [Component Diagrams](#6-component-diagrams)

---

## 1. Architecture Overview

### 1.1. System Architecture

```plantuml
@startuml RAGFlow_Architecture
!define RECTANGLE class

skinparam backgroundColor #FEFEFE
skinparam componentStyle rectangle

package "Frontend Layer" {
    [React SPA] as Frontend
    [Ant Design] as AntD
    [Zustand Store] as Store
}

package "API Gateway Layer" {
    [Quart Server] as API
    [CORS Middleware] as CORS
    [JWT Auth] as Auth
    [Blueprint Router] as Router
}

package "Service Layer" {
    [DocumentService] as DocSvc
    [DialogService] as DialogSvc
    [KnowledgebaseService] as KBSvc
    [ConversationService] as ConvSvc
    [CanvasService] as CanvasSvc
    [LLMService] as LLMSvc
}

package "RAG Engine" {
    [Query Processor] as QueryProc
    [Embedding Model] as EmbedModel
    [Rerank Model] as Rerank
    [Prompt Generator] as PromptGen
}

package "Agent System" {
    [Canvas Executor] as Canvas
    [Component Base] as CompBase
    [Tool Manager] as ToolMgr
}

package "Processing Layer" {
    [Task Executor] as TaskExec
    [Document Parser] as Parser
    [Chunker] as Chunk
    [Tokenizer] as Token
}

package "Data Layer" {
    database "MySQL/PostgreSQL" as DB
    database "Elasticsearch/Infinity" as ES
    database "Redis" as Redis
    storage "MinIO/S3" as Storage
}

' Frontend connections
Frontend --> API : HTTP/SSE
Frontend ..> AntD : uses
Frontend ..> Store : state

' API Layer connections
API --> Router : route
API --> Auth : authenticate
API --> CORS : handle

' Router to Services
Router --> DocSvc
Router --> DialogSvc
Router --> KBSvc
Router --> ConvSvc
Router --> CanvasSvc

' Service to RAG
DialogSvc --> QueryProc
DialogSvc --> LLMSvc
QueryProc --> EmbedModel
DialogSvc --> Rerank
DialogSvc --> PromptGen

' Service to Agent
CanvasSvc --> Canvas
Canvas --> CompBase
Canvas --> ToolMgr

' Service to Processing
DocSvc --> TaskExec
TaskExec --> Parser
TaskExec --> Chunk
TaskExec --> Token

' All services to Data
DocSvc --> DB
DialogSvc --> DB
KBSvc --> DB
ConvSvc --> DB
CanvasSvc --> DB

QueryProc --> ES
EmbedModel --> ES
DocSvc --> Storage
TaskExec --> Redis

note right of API
  Async Quart Framework
  REST + SSE endpoints
end note

note right of RAG Engine
  Hybrid Search:
  - Vector similarity
  - BM25 full-text
  - Reranking
end note

note right of Agent System
  Component-based
  workflow execution
end note

@enduml
```

### 1.2. Deployment Architecture

```plantuml
@startuml RAGFlow_Deployment
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Deployment.puml

Deployment_Node(client, "Client", "Browser/Mobile") {
    Container(web, "Web UI", "React/TypeScript", "User interface")
}

Deployment_Node(docker, "Docker Host", "Linux") {
    Deployment_Node(nginx, "Nginx", "Reverse Proxy") {
        Container(proxy, "Nginx", "1.25", "Load balancer & SSL")
    }

    Deployment_Node(app, "Application Container", "Python 3.10") {
        Container(ragflow, "RAGFlow Server", "Quart", "Main API server")
        Container(task_worker, "Task Worker", "Trio", "Background processing")
    }

    Deployment_Node(data, "Data Services") {
        ContainerDb(mysql, "MySQL", "8.0", "Metadata storage")
        ContainerDb(es, "Elasticsearch", "8.x", "Vector & full-text search")
        ContainerDb(redis, "Redis", "7.x", "Cache & queue")
        ContainerDb(minio, "MinIO", "Latest", "Object storage")
    }
}

Rel(web, proxy, "HTTPS", "443")
Rel(proxy, ragflow, "HTTP", "9380")
Rel(ragflow, task_worker, "Redis Queue", "Async tasks")
Rel(ragflow, mysql, "SQL", "3306")
Rel(ragflow, es, "REST API", "9200")
Rel(ragflow, redis, "Protocol", "6379")
Rel(ragflow, minio, "S3 API", "9000")
Rel(task_worker, mysql, "SQL", "3306")
Rel(task_worker, es, "REST API", "9200")
Rel(task_worker, minio, "S3 API", "9000")

@enduml
```

---

## 2. Document Processing Flow

### 2.1. Upload & Parsing Sequence

```plantuml
@startuml Document_Upload_Flow
!theme plain
skinparam sequenceMessageAlign center
skinparam responseMessageBelowArrow true

actor User
participant "Web UI" as UI
participant "document_app.py" as API
participant "FileService" as FS
participant "MinIO" as Storage
participant "DocumentService" as DS
participant "TaskService" as TS
participant "Redis Queue" as Redis
participant "TaskExecutor" as TE
participant "Parser" as Parser
participant "Chunker" as Chunker
participant "EmbeddingModel" as Embed
participant "Elasticsearch" as ES

User -> UI: Upload file
UI -> API: POST /v1/document/upload\n{kb_id, file}

activate API
API -> API: Validate file type\nCheck KB permission
API -> FS: upload_document(kb, file)

activate FS
FS -> FS: Validate file:\n- Type in allowed list\n- Size < limit\n- Name length < 255

FS -> Storage: PUT /bucket/kb_id/file
activate Storage
Storage --> FS: Storage location
deactivate Storage

FS -> DS: insert(document)
activate DS
DS -> DS: Create Document record:\n- id, kb_id, parser_id\n- type, name, location\n- parser_config
DS --> FS: Document object
deactivate DS

FS -> FS: Generate thumbnail
FS --> API: [doc1, doc2, ...]
deactivate FS

API -> TS: create_tasks(docs)
activate TS
TS -> TS: Split doc into page ranges\n(e.g., 0-10, 10-20, ...)
loop For each page range
    TS -> Redis: LPUSH task_queue\n{doc_id, from_page, to_page}
end
TS --> API: Tasks created
deactivate TS

API --> UI: {docs: [{id, name, status}]}
deactivate API

UI --> User: Show upload success

' Background processing
... Background Task Processing ...

Redis -> TE: RPOP task_queue
activate TE

TE -> TE: Acquire semaphore\n(max 5 concurrent)

TE -> DS: get_by_id(doc_id)
DS --> TE: Document + parser_config

TE -> Storage: GET /bucket/location
Storage --> TE: File blob

TE -> Parser: parse(blob, parser_type)
activate Parser

alt PDF
    Parser -> Parser: deepdoc.PDFParser:\n- Extract text\n- OCR if needed\n- Layout analysis
else Image
    Parser -> Parser: Vision + OCR:\n- Tesseract/PaddleOCR\n- Layout detection
else Office
    Parser -> Parser: python-docx/openpyxl:\n- Extract content\n- Preserve structure
end

Parser --> TE: Extracted text + metadata
deactivate Parser

TE -> Chunker: split(text, config)
activate Chunker
Chunker -> Chunker: Apply chunking strategy:\n- Naive: fixed size\n- QA: question-answer\n- Table: preserve tables
Chunker --> TE: chunks[]
deactivate Chunker

TE -> TE: Tokenize chunks
TE -> Embed: encode(chunks)
activate Embed
Embed -> Embed: Generate embeddings:\n- Batch processing\n- Normalize vectors
Embed --> TE: vectors[]
deactivate Embed

TE -> ES: bulk_index(chunks)
activate ES
ES -> ES: Store:\n- Vector field (dense_vector)\n- Full-text fields (text)\n- Metadata (doc_id, kb_id)
ES --> TE: Index response
deactivate ES

TE -> DS: update_progress(doc_id, 100%)
DS --> TE: Updated

TE -> Redis: ACK task
deactivate TE

@enduml
```

### 2.2. Task Processing State Machine

```plantuml
@startuml Task_State_Machine
!theme plain

[*] --> PENDING : Task created

PENDING --> RUNNING : Worker picks up task
RUNNING --> DONE : Processing complete
RUNNING --> FAILED : Error occurred
RUNNING --> CANCEL : User cancels

FAILED --> RUNNING : Retry (max 3)
FAILED --> [*] : Max retries reached

DONE --> [*]
CANCEL --> [*]

state RUNNING {
    [*] --> Parsing
    Parsing --> Chunking : Text extracted
    Chunking --> Embedding : Chunks created
    Embedding --> Indexing : Vectors generated
    Indexing --> [*] : Complete

    Parsing --> [*] : Error
    Chunking --> [*] : Error
    Embedding --> [*] : Error
    Indexing --> [*] : Error
}

note right of RUNNING
  Progress tracking:
  - 0-30%: Parsing
  - 30-60%: Chunking
  - 60-90%: Embedding
  - 90-100%: Indexing
end note

@enduml
```

---

## 3. RAG Retrieval Flow

### 3.1. Hybrid Search Sequence

```plantuml
@startuml RAG_Retrieval_Flow
!theme plain
skinparam sequenceMessageAlign center

actor User
participant "Chat UI" as UI
participant "conversation_app.py" as API
participant "DialogService" as DialogSvc
participant "Query Processor" as QP
participant "EmbeddingModel" as Embed
participant "Elasticsearch" as ES
participant "RerankModel" as Rerank
participant "PromptGenerator" as PromptGen
participant "ChatModel" as LLM

User -> UI: Ask question
UI -> API: POST /v1/conversation/completion\n{conversation_id, messages}

activate API
API -> DialogSvc: get_dialog_config(dialog_id)
DialogSvc --> API: Dialog config:\n- kb_ids\n- similarity_threshold\n- top_k, rerank_id

API -> DialogSvc: chat(dialog, messages, stream=True)
activate DialogSvc

' Retrieval phase
group Retrieval Phase
    DialogSvc -> QP: process_query(question)
    activate QP

    QP -> QP: Tokenize:\n"What is RAG?"\n→ ["what", "rag"]

    QP -> QP: Extract keywords:\nRemove stopwords\nStem words

    QP --> DialogSvc: keywords + query_clauses
    deactivate QP

    DialogSvc -> Embed: encode_queries([question])
    activate Embed
    Embed -> Embed: Model forward pass:\ntext → 768-dim vector
    Embed --> DialogSvc: query_vector
    deactivate Embed

    DialogSvc -> ES: hybrid_search(\nquery_vector,\nkeywords,\nkb_ids,\ntop_k=100)
    activate ES

    ES -> ES: Score fusion:\nvector_score * 0.3 +\nbm25_score * 0.7

    ES -> ES: Apply filters:\n- kb_id in [kb1, kb2]\n- available_int = 1\n- similarity > threshold

    ES --> DialogSvc: Top 100 chunks
    deactivate ES

    alt Reranking enabled
        DialogSvc -> Rerank: rerank(query, chunks)
        activate Rerank

        Rerank -> Rerank: Cross-encoder:\nScore each (query, chunk) pair

        Rerank -> Rerank: Sort by relevance
        Rerank -> Rerank: Take top_k (e.g., 6)

        Rerank --> DialogSvc: Reordered chunks
        deactivate Rerank
    else No reranking
        DialogSvc -> DialogSvc: Take top_k directly
    end
end

' Generation phase
group Generation Phase
    DialogSvc -> PromptGen: construct_prompt(\nquestion,\nchunks,\nconv_history)
    activate PromptGen

    PromptGen -> PromptGen: Format system prompt:\n"You are an assistant..."

    PromptGen -> PromptGen: Format knowledge:\n"[ID:0] chunk1\n[ID:1] chunk2"

    PromptGen -> PromptGen: Add citation instructions:\n"Cite using [ID:N]"

    PromptGen --> DialogSvc: final_prompt
    deactivate PromptGen

    DialogSvc -> LLM: chat_streamly(\nsystem_prompt,\nmessages,\ngen_config)
    activate LLM

    loop Streaming tokens
        LLM --> DialogSvc: token_chunk
        DialogSvc -> DialogSvc: Accumulate answer
        DialogSvc -> DialogSvc: Parse citations:\nExtract [ID:N]
        DialogSvc --> API: yield {\n  answer,\n  reference: {chunks}\n}
        API --> UI: SSE: data: {...}
        UI --> User: Display incremental answer
    end

    LLM --> DialogSvc: Complete
    deactivate LLM
end

DialogSvc -> DialogSvc: Extract cited chunks:\nOnly chunks referenced in answer

DialogSvc --> API: Final result
deactivate DialogSvc

API -> API: Save conversation:\n- message history\n- reference chunks

API --> UI: SSE: data: true (end)
deactivate API

@enduml
```

### 3.2. Elasticsearch Query Structure

```plantuml
@startuml ES_Query_Structure
!theme plain

package "Elasticsearch Query" {
    card "bool" as bool {
        card "must" as must {
            label "Filter clauses" as f1
            rectangle "terms: {kb_id: [kb1, kb2]}" as kb
            rectangle "term: {available_int: 1}" as avail
        }

        card "should" as should {
            label "Scoring clauses" as s1

            card "script_score" as script {
                label "Vector Similarity" as vs
                rectangle "cosineSimilarity(\n  params.query_vector,\n  'q_vec'\n) + 1.0" as cosine
            }

            card "multi_match" as mm {
                label "Full-text BM25" as bm25
                rectangle "fields: [\n  'content_tks^1',\n  'title_tks^8'\n]" as fields
                rectangle "query: 'extracted keywords'" as query
            }
        }

        card "filter" as filter {
            label "Post-filter" as pf
            rectangle "range: {\n  vector_similarity: {gte: 0.2}\n}" as sim
        }
    }
}

must -down-> kb
must -down-> avail
should -down-> script
should -down-> mm
filter -down-> sim

script -down-> cosine
mm -down-> fields
mm -down-> query

note right of script
  Vector score: 0.0 - 2.0
  Weight: 0.3 (configurable)
end note

note right of mm
  BM25 score: varies
  Weight: 0.7 (configurable)

  Title boost: 8x
  Content boost: 1x
end note

note bottom of bool
  Final score =
    vector_score * 0.3 +
    bm25_score * 0.7

  Then filter by threshold
end note

@enduml
```

---

## 4. Chat Conversation Flow

### 4.1. Complete Conversation Lifecycle

```plantuml
@startuml Conversation_Lifecycle
!theme plain

participant "User" as User
participant "Frontend" as UI
participant "conversation_app.py" as API
participant "ConversationService" as ConvSvc
participant "DialogService" as DialogSvc
participant "RAG Engine" as RAG
participant "LLMBundle" as LLM
participant "Database" as DB

== Conversation Creation ==

User -> UI: Click "New Chat"
UI -> API: POST /v1/conversation/set\n{dialog_id, is_new: true}

activate API
API -> DialogSvc: get_dialog(dialog_id)
DialogSvc -> DB: SELECT * FROM dialog
DB --> DialogSvc: Dialog config

API -> ConvSvc: create_conversation(\ndialog_id,\nuser_id)
activate ConvSvc

ConvSvc -> ConvSvc: Generate conversation:\n- id = uuid\n- name = "New conversation"\n- message = [prologue]\n- reference = []

ConvSvc -> DB: INSERT INTO conversation
DB --> ConvSvc: Success

ConvSvc --> API: Conversation object
deactivate ConvSvc

API --> UI: {id, name, message, reference}
deactivate API
UI --> User: Show chat interface

== Message Exchange ==

User -> UI: Type question:\n"What is RAG?"
UI -> API: POST /v1/conversation/completion\n{\n  conversation_id,\n  messages: [\n    {role: "user", content: "What is RAG?"}\n  ],\n  stream: true\n}

activate API

API -> ConvSvc: get_conversation(conv_id)
ConvSvc -> DB: SELECT * FROM conversation
DB --> ConvSvc: Conversation + history

API -> DialogSvc: chat(dialog, messages, stream=True)
activate DialogSvc

alt Has KB IDs
    DialogSvc -> RAG: retrieve(question, kb_ids)
    activate RAG
    RAG -> RAG: Hybrid search\n+ Reranking
    RAG --> DialogSvc: chunks[]
    deactivate RAG

    DialogSvc -> DialogSvc: Format prompt:\nSystem + Knowledge + Question
else Chat-only mode
    DialogSvc -> DialogSvc: Format prompt:\nSystem + Question
end

DialogSvc -> LLM: chat_streamly(prompt, config)
activate LLM

loop Stream response
    LLM --> DialogSvc: token
    DialogSvc -> DialogSvc: Accumulate:\n"RAG stands for..."
    DialogSvc -> DialogSvc: Parse citations:\n[ID:0], [ID:1]
    DialogSvc --> API: yield {\n  answer: partial_answer,\n  reference: {chunks, doc_aggs}\n}
    API --> UI: SSE: data: {...}
    UI --> User: Update UI incrementally
end

LLM --> DialogSvc: <END>
deactivate LLM

DialogSvc --> API: Final answer
deactivate DialogSvc

API -> ConvSvc: update_conversation(\nconv_id,\nnew_messages,\nreferences)
activate ConvSvc

ConvSvc -> ConvSvc: Append to history:\nmessages.append(user_msg)\nmessages.append(assistant_msg)\nreference.append(chunks)

ConvSvc -> DB: UPDATE conversation\nSET message = ?,\n    reference = ?,\n    update_time = ?
DB --> ConvSvc: Success

ConvSvc --> API: Updated
deactivate ConvSvc

API --> UI: SSE: data: true
deactivate API

UI --> User: Show complete answer

== Conversation History ==

User -> UI: View history
UI -> API: GET /v1/conversation/get\n?conversation_id={id}

activate API
API -> ConvSvc: get_by_id(conv_id)
ConvSvc -> DB: SELECT * FROM conversation
DB --> ConvSvc: Full conversation

ConvSvc -> ConvSvc: Format references:\nchunks_format(ref)

ConvSvc --> API: {\n  id, name,\n  message: [...],\n  reference: [...]\n}
deactivate API

API --> UI: Conversation object
UI --> User: Display chat history

@enduml
```

### 4.2. Prompt Construction Flow

```plantuml
@startuml Prompt_Construction
!theme plain

start

:Receive inputs:
- question
- conv_history
- chunks
- dialog_config;

:Load prompt template;
note right
  Templates in rag/prompts/
  Using Jinja2 engine
end note

partition "Build System Prompt" {
    :Get base system prompt;

    if (Has custom system?) then (yes)
        :Use dialog.prompt_config.system;
    else (no)
        :Use default template;
    endif

    if (Cross-language?) then (yes)
        :Add language instruction:
        "Please answer in {language}";
    endif
}

partition "Build Knowledge Context" {
    if (Has chunks?) then (yes)
        :Format chunks with IDs;
        note right
          [ID:0] chunk1 content
          [ID:1] chunk2 content
          [ID:2] chunk3 content
        end note

        :Add citation instructions;
        note right
          "Please cite sources using [ID:N]
          format in your answer"
        end note
    else (no)
        :knowledge_context = "";
    endif
}

partition "Build Conversation History" {
    :Format previous messages;

    repeat
        :message = history[i];

        if (message.role == "user") then
            :Add "User: {content}";
        else
            :Add "Assistant: {content}";
        endif

    repeat while (More messages?)
}

partition "Build Current Question" {
    :Format question;

    if (Has conversation context?) then (yes)
        :full_question =
        "Based on our previous conversation:
        {history_summary}

        Now answer: {question}";
    else (no)
        :full_question = question;
    endif
}

partition "Assemble Final Prompt" {
    :Combine all parts:

    final_prompt =
    {system_prompt}

    {knowledge_context}

    {conversation_history}

    User: {full_question}
    Assistant:;

    :Check token limit;

    if (tokens > max_tokens) then (yes)
        :Truncate:
        - Trim old history
        - Keep recent context
        - Always keep question;
    endif
}

:Return final_prompt;

stop

@enduml
```

---

## 5. Agent Workflow Execution

### 5.1. Canvas Execution Flow

```plantuml
@startuml Agent_Canvas_Execution
!theme plain
skinparam sequenceMessageAlign center

actor User
participant "Web UI" as UI
participant "canvas_app.py" as API
participant "CanvasService" as CanvasSvc
participant "Canvas/Graph" as Canvas
participant "Begin" as Begin
participant "LLM Component" as LLMComp
participant "Retrieval" as RetrievalComp
participant "Categorize" as CategorizeComp
participant "ChatModel" as LLM
participant "DialogService" as DialogSvc
participant "Redis" as Redis

User -> UI: Design workflow in canvas
UI -> API: POST /v1/canvas/set\n{dsl, title}

activate API
API -> CanvasSvc: save(dsl, title, user_id)
CanvasSvc --> API: canvas_id
API --> UI: {id, dsl}
deactivate API

User -> UI: Click "Run"
UI -> API: POST /v1/canvas/completion\n{\n  canvas_id,\n  question: "What is RAG?"\n}

activate API
API -> CanvasSvc: get(canvas_id)
CanvasSvc --> API: Canvas DSL

API -> Canvas: Canvas.run(\nquestion="What is RAG?",\ntenant_id=user_id)
activate Canvas

Canvas -> Canvas: Parse DSL:\n- Load components\n- Build execution graph\n- Validate connections

Canvas -> Canvas: Initialize globals:\n{\n  sys.query: "What is RAG?",\n  sys.user_id: user_id,\n  sys.conversation_turns: 0\n}

Canvas -> Begin: invoke(**inputs)
activate Begin

Begin -> Begin: Validate inputs:\n- Check required fields\n- Set default values

Begin -> Begin: Set initial state:\noutputs = {\n  "query": "What is RAG?",\n  "user_id": user_id\n}

Begin --> Canvas: {\n  outputs,\n  downstream: ["llm_0"]\n}
deactivate Begin

Canvas -> Canvas: callback("Started", 10%)
Canvas --> API: Progress update
API --> UI: SSE: {component: "begin", progress: 10%}

Canvas -> LLMComp: invoke(**outputs)
activate LLMComp

LLMComp -> LLMComp: Render prompt template:\nreplace {query} with "What is RAG?"

LLMComp -> LLM: chat_streamly(\nprompt="Analyze: What is RAG?",\nconfig={temperature: 0.7})
activate LLM

loop Stream response
    LLM --> LLMComp: token
    LLMComp -> Canvas: callback(token, 40%)
    Canvas --> API: Progress update
    API --> UI: SSE: {component: "llm_0", text: token}
end

LLM --> LLMComp: Complete:\n"This is a question about\nRetrieval-Augmented Generation"
deactivate LLM

LLMComp --> Canvas: {\n  outputs: {analysis: "..."},\n  downstream: ["categorize_0"]\n}
deactivate LLMComp

Canvas -> CategorizeComp: invoke(**outputs)
activate CategorizeComp

CategorizeComp -> CategorizeComp: Classify input:\nCategories: [technical, general]

CategorizeComp -> LLM: chat(\nprompt="Classify: {analysis}",\nconfig={temperature: 0.1})
activate LLM
LLM --> CategorizeComp: "technical"
deactivate LLM

CategorizeComp -> CategorizeComp: Route based on category:\ndownstream_map = {\n  "technical": "retrieval_0",\n  "general": "answer_0"\n}

CategorizeComp --> Canvas: {\n  outputs: {category: "technical"},\n  downstream: ["retrieval_0"]\n}
deactivate CategorizeComp

Canvas -> Canvas: callback("Categorized", 60%)
Canvas --> API: Progress
API --> UI: SSE update

Canvas -> RetrievalComp: invoke(**outputs)
activate RetrievalComp

RetrievalComp -> DialogSvc: ask(\nquestion="What is RAG?",\nkb_ids=["kb_123"],\ntop_k=5)
activate DialogSvc

DialogSvc -> DialogSvc: Hybrid search\n+ Reranking

DialogSvc --> RetrievalComp: chunks[]
deactivate DialogSvc

RetrievalComp -> RetrievalComp: Format chunks:\nchunks_format(chunks)

RetrievalComp --> Canvas: {\n  outputs: {\n    chunks: [...],\n    formatted_text: "..."\n  },\n  downstream: ["answer_0"]\n}
deactivate RetrievalComp

Canvas -> Canvas: callback("Retrieved", 80%)
Canvas --> API: Progress
API --> UI: SSE update

Canvas -> Canvas: Execute Answer component
Canvas -> Canvas: Collect final outputs

Canvas -> Redis: Store execution trace:\nLPUSH canvas:{id}:trace\n{component logs}
Redis --> Canvas: OK

Canvas --> API: {\n  answer: final_answer,\n  reference: chunks,\n  trace: execution_log\n}
deactivate Canvas

API --> UI: SSE: data: {result}
deactivate API

UI --> User: Display final answer

@enduml
```

### 5.2. Component Interaction Diagram

```plantuml
@startuml Component_Interactions
!theme plain

interface ComponentBase {
    + invoke(**kwargs)
    + callback(message, progress)
}

class Begin implements ComponentBase {
    - params: dict
    + invoke(**kwargs)
}

class LLM implements ComponentBase {
    - llm_id: str
    - prompt: str
    - temperature: float
    + invoke(**kwargs)
    - render_prompt(template, vars)
}

class Retrieval implements ComponentBase {
    - kb_ids: list
    - top_k: int
    - similarity_threshold: float
    + invoke(query, **kwargs)
}

class Categorize implements ComponentBase {
    - categories: list
    - downstream_map: dict
    + invoke(input, **kwargs)
}

class Switch implements ComponentBase {
    - conditions: list
    + invoke(**kwargs)
    - evaluate_condition(cond, vars)
}

class Iteration implements ComponentBase {
    - max_iterations: int
    - loop_body: list
    + invoke(**kwargs)
}

class Answer implements ComponentBase {
    + invoke(**kwargs)
}

class Canvas {
    - dsl: dict
    - components: dict
    - tenant_id: str
    + run(**kwargs)
    + on_progress(comp_id, msg, progress)
    - build_graph()
    - execute_component(comp_id, inputs)
}

Canvas *-- ComponentBase : manages
Canvas ..> Begin : executes
Canvas ..> LLM : executes
Canvas ..> Retrieval : executes
Canvas ..> Categorize : executes

LLM ..> "LLMBundle" : uses
Retrieval ..> "DialogService" : uses
Categorize ..> "LLMBundle" : uses

note right of Canvas
  Execution flow:
  1. Parse DSL
  2. Build graph
  3. Execute from Begin
  4. Follow downstream edges
  5. Collect outputs
end note

note bottom of ComponentBase
  All components implement:
  - invoke(): Main logic
  - callback(): Progress reporting

  Component params from DSL
end note

@enduml
```

### 5.3. Tool Integration Architecture

```plantuml
@startuml Tool_Architecture
!theme plain

package "Agent System" {
    class ToolManager {
        - tools: dict
        + register_tool(name, tool)
        + execute(tool_name, **params)
    }

    interface ToolBase {
        + execute(**params)
        + validate_params(params)
    }
}

package "Built-in Tools" {
    class RetrievalTool implements ToolBase {
        - dialog_service: DialogService
        + execute(query, kb_ids)
    }

    class TavilyTool implements ToolBase {
        - api_key: str
        + execute(query, max_results)
    }

    class WikipediaTool implements ToolBase {
        + execute(query, lang)
    }

    class CodeExecTool implements ToolBase {
        - sandbox: bool
        + execute(code, language)
    }

    class SQLTool implements ToolBase {
        - connection: str
        + execute(query)
    }
}

package "External APIs" {
    cloud "Tavily API" as Tavily
    cloud "Wikipedia API" as Wiki
    database "SQL Database" as SQL
    component "Code Sandbox" as Sandbox
}

package "RAG Components" {
    class DialogService {
        + ask(question, kb_ids)
    }
}

ToolManager o-- ToolBase : manages
ToolManager ..> RetrievalTool : uses
ToolManager ..> TavilyTool : uses
ToolManager ..> WikipediaTool : uses
ToolManager ..> CodeExecTool : uses
ToolManager ..> SQLTool : uses

RetrievalTool --> DialogService : calls
TavilyTool --> Tavily : HTTP request
WikipediaTool --> Wiki : HTTP request
CodeExecTool --> Sandbox : execute
SQLTool --> SQL : query

note right of ToolManager
  Tools are registered at startup:

  tools = {
    "retrieval": RetrievalTool(),
    "tavily": TavilyTool(),
    "wikipedia": WikipediaTool(),
    ...
  }
end note

note bottom of ToolBase
  Tool execution pattern:
  1. Validate params
  2. Execute operation
  3. Return structured result
  4. Handle errors gracefully
end note

@enduml
```

---

## 6. Component Diagrams

### 6.1. Service Layer Components

```plantuml
@startuml Service_Layer
!theme plain

package "API Layer" {
    [conversation_app.py] as ConvApp
    [document_app.py] as DocApp
    [canvas_app.py] as CanvasApp
    [dialog_app.py] as DialogApp
}

package "Service Layer" {
    component "ConversationService" as ConvSvc {
        [get_by_id()]
        [update()]
        [save()]
    }

    component "DocumentService" as DocSvc {
        [insert()]
        [update_progress()]
        [get_by_kb_id()]
    }

    component "DialogService" as DialogSvc {
        [chat()]
        [ask()]
        [gen_mindmap()]
    }

    component "CanvasService" as CanvasSvc {
        [save()]
        [run()]
        [get()]
    }

    component "LLMService" as LLMSvc {
        [chat()]
        [chat_streamly()]
        [encode()]
    }

    component "KnowledgebaseService" as KBSvc {
        [get_by_ids()]
        [update()]
    }
}

package "Data Access" {
    database "MySQL/PostgreSQL" as DB
    database "Elasticsearch" as ES
    database "Redis" as Redis
}

ConvApp --> ConvSvc
DocApp --> DocSvc
CanvasApp --> CanvasSvc
DialogApp --> DialogSvc

DialogSvc --> LLMSvc
DialogSvc --> KBSvc
CanvasSvc --> DialogSvc

ConvSvc --> DB
DocSvc --> DB
DialogSvc --> DB
CanvasSvc --> DB
KBSvc --> DB

DialogSvc --> ES
LLMSvc --> ES
DocSvc --> Redis

@enduml
```

### 6.2. RAG Engine Components

```plantuml
@startuml RAG_Engine
!theme plain

package "RAG Engine" {
    component "Query Processor" {
        [FulltextQueryer]
        [Tokenizer]
        [KeywordExtractor]
    }

    component "Search Engine" {
        [Dealer]
        [DocStoreConnection]
        [QueryBuilder]
    }

    component "Model Layer" {
        [EmbeddingModel]
        [ChatModel]
        [RerankModel]
        [Image2TextModel]
        [TTSModel]
    }

    component "Prompt Engine" {
        [PromptGenerator]
        [TemplateLoader]
        [ChunkFormatter]
    }
}

package "External Services" {
    cloud "LLM APIs" {
        [OpenAI]
        [Azure OpenAI]
        [Anthropic]
        [Ollama]
        [Xinference]
    }

    database "Vector Store" {
        [Elasticsearch]
        [Infinity]
        [OpenSearch]
    }
}

[FulltextQueryer] --> [Tokenizer]
[Dealer] --> [DocStoreConnection]
[DocStoreConnection] --> [Elasticsearch]
[DocStoreConnection] --> [Infinity]

[EmbeddingModel] --> [OpenAI]
[EmbeddingModel] --> [Ollama]
[ChatModel] --> [OpenAI]
[ChatModel] --> [Anthropic]
[RerankModel] --> [OpenAI]

[PromptGenerator] --> [TemplateLoader]
[PromptGenerator] --> [ChunkFormatter]

note right of "Model Layer"
  LiteLLM Integration:
  - 30+ LLM providers
  - Unified API interface
  - Automatic retry
  - Load balancing
end note

@enduml
```

### 6.3. Database Schema

```plantuml
@startuml Database_Schema
!theme plain

entity "Tenant" as tenant {
    * id: varchar(32) <<PK>>
    --
    name: varchar(255)
    llm_id: varchar(255)
    embd_id: varchar(255)
    rerank_id: varchar(255)
    tts_id: varchar(255)
    asr_id: varchar(255)
    create_time: bigint
    update_time: bigint
}

entity "User" as user {
    * id: varchar(32) <<PK>>
    --
    tenant_id: varchar(32) <<FK>>
    email: varchar(255)
    nickname: varchar(255)
    access_token: varchar(255)
    status: varchar(1)
    create_time: bigint
}

entity "Knowledgebase" as kb {
    * id: varchar(32) <<PK>>
    --
    tenant_id: varchar(32) <<FK>>
    name: varchar(255)
    avatar: varchar(255)
    description: text
    language: varchar(32)
    embd_id: varchar(255)
    parser_id: varchar(32)
    parser_config: text
    chunk_num: int
    token_num: bigint
    create_time: bigint
}

entity "Document" as doc {
    * id: varchar(32) <<PK>>
    --
    kb_id: varchar(32) <<FK>>
    parser_id: varchar(32)
    parser_config: text
    name: varchar(255)
    type: varchar(32)
    location: varchar(255)
    size: bigint
    thumbnail: text
    progress: float
    progress_msg: text
    run: varchar(16)
    chunk_num: int
    token_num: bigint
    create_time: bigint
}

entity "Task" as task {
    * id: varchar(32) <<PK>>
    --
    doc_id: varchar(32) <<FK>>
    from_page: int
    to_page: int
    progress: float
    progress_msg: text
    retry_count: int
    chunk_ids: text
    create_time: bigint
}

entity "Dialog" as dialog {
    * id: varchar(32) <<PK>>
    --
    tenant_id: varchar(32) <<FK>>
    name: varchar(255)
    description: text
    icon: varchar(255)
    kb_ids: text
    llm_id: varchar(255)
    llm_setting: text
    prompt_type: varchar(32)
    prompt_config: text
    similarity_threshold: float
    vector_similarity_weight: float
    top_n: int
    top_k: int
    rerank_id: varchar(255)
    create_time: bigint
}

entity "Conversation" as conv {
    * id: varchar(32) <<PK>>
    --
    dialog_id: varchar(32) <<FK>>
    user_id: varchar(32) <<FK>>
    name: varchar(255)
    message: text
    reference: text
    create_time: bigint
    update_time: bigint
}

entity "UserCanvas" as canvas {
    * id: varchar(32) <<PK>>
    --
    tenant_id: varchar(32) <<FK>>
    title: varchar(255)
    dsl: text
    canvas_category: varchar(32)
    create_time: bigint
}

tenant ||--o{ user
tenant ||--o{ kb
tenant ||--o{ dialog
tenant ||--o{ canvas

kb ||--o{ doc
doc ||--o{ task

dialog ||--o{ conv
user ||--o{ conv

note right of kb
  Knowledge base metadata:
  - Parser configuration
  - Embedding model ID
  - Statistics (chunk/token count)
end note

note right of doc
  Document states:
  - UNSTART: Not processed
  - RUNNING: Processing
  - DONE: Complete
  - CANCEL: Cancelled
  - FAILED: Error
end note

note right of dialog
  Dialog configuration:
  - KB associations
  - LLM settings
  - Prompt templates
  - Retrieval parameters
end note

note right of conv
  Conversation data:
  - message: JSON array of messages
  - reference: JSON array of cited chunks
end note

@enduml
```

### 6.4. Elasticsearch Index Structure

```plantuml
@startuml ES_Index_Structure
!theme plain

package "Elasticsearch Index: ragflow_{tenant_id}" {
    map "Document Mapping" {
        doc_id => keyword
        kb_id => keyword
        docnm_kwd => keyword
        title_tks => text (tokenized)
        content_ltks => text (tokenized)
        content_sm_ltks => text (fine-grained)
        q_vec => dense_vector (768/1024 dim)
        available_int => integer
        create_timestamp_flt => float
        kb_id_kwd => keyword
        img_id => keyword
        page_num_int => integer
        position_int => text
    }

    note right of "Document Mapping"
        Vector field:
        - Dimension: 768 (BERT-style)
                    or 1024 (large models)
        - Similarity: cosine
        - Index: true (for KNN search)

        Full-text fields:
        - title_tks: Title tokens (8x boost)
        - content_ltks: Content tokens
        - content_sm_ltks: Fine-grained (synonyms)

        Filter fields:
        - kb_id: Knowledge base filter
        - available_int: 0/1 status
        - doc_id: Document filter
    end note
}

package "Index Settings" {
    card "Analysis" {
        label "Tokenizers" as tok
        rectangle "rag_tokenizer: Custom tokenizer\n- Language-specific\n- Stop words removal\n- Stemming" as custom_tok

        rectangle "ik_max_word: Chinese\nstandard: English" as std_tok
    }

    card "Similarity" {
        label "Scoring" as score
        rectangle "BM25 for full-text\nCosine for vectors" as sim
    }

    card "Shards" {
        label "Distribution" as dist
        rectangle "Shards: 1 (default)\nReplicas: 0 (dev)\nReplicas: 1+ (prod)" as shard
    }
}

note bottom of "Index Settings"
    Index naming pattern:
    - ragflow_{tenant_id}
    - One index per tenant
    - Multiple KB per index

    Query types:
    1. Vector KNN search
    2. BM25 full-text
    3. Hybrid (combined)
    4. Filtered search
end note

@enduml
```

---

## 📝 Notes

### Viewing PlantUML Diagrams

**Online Viewers:**
- PlantUML Online Server: http://www.plantuml.com/plantuml/uml/
- PlantText: https://www.planttext.com/

**VS Code Extension:**
```bash
# Install PlantUML extension
code --install-extension jebbs.plantuml

# Or search "PlantUML" in Extensions marketplace
```

**Command Line:**
```bash
# Install PlantUML
brew install plantuml  # macOS
apt install plantuml   # Ubuntu

# Generate diagrams
plantuml diagram.puml
# Output: diagram.png
```

### Diagram Types Included

✅ Sequence Diagrams - Flow of operations
✅ Component Diagrams - System structure
✅ Deployment Diagrams - Infrastructure
✅ State Machine - Task states
✅ Class Diagrams - Object relationships
✅ Entity Relationship - Database schema

---

**Generated:** 2025-11-23
**Format:** PlantUML 1.2024.x compatible
