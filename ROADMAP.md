# Roadmap

## Conceptual Phases

### Phase 1 - Local AI
Foundations of running a language model locally: LLMs, inference runtimes, tokenizers, generation parameters, model size, quantization, and performance measurement.

### Phase 2 - Retrieval
Progress from fundamental similarity mechanisms to concrete vector stores:
- Embeddings and their meaning
- Vector similarity metrics (dot product, cosine, Euclidean)
- Manual/NumPy similarity experiments to understand the math
- Chunking strategies for documents
- FAISS for efficient similarity search
- Persistent vector stores (e.g., saving/loading indexes)
- Basic retrieval pipelines combining the above

### Phase 3 - RAG
Combining retrieval with generation: simple RAG, multi‑document RAG, persistent RAG, metadata filtering, retrieval evaluation, and observability.

### Phase 4 - RAG and AI Security
Security‑focused experiments on the RAG pipeline: prompt injection, indirect injection, malicious documents, RAG poisoning, embedding attacks, data leakage, metadata/ACL attacks, and provenance.

### Phase 5 - Model Adaptation
Techniques for adapting models locally: fine‑tuning, LoRA, QLoRA, and other parameter‑efficient methods.

### Phase 6 - Agentic AI and MCP
Building agents that use tools, integrating the Model Context Protocol (MCP), agentic RAG, and securing agentic systems.

## Initial Lab Progression (Working Roadmap)

**Lab01** – Running a Local LLM  
**Lab02** – Understanding the Inference Runtime  
**Lab03** – Tokens, Context and Generation Parameters  
**Lab04** – Model Size and Quantization  
**Lab05** – Measuring Local AI Performance  

Following labs will progress into:
- Embeddings and similarity search
- Chunking strategies
- FAISS vector store
- Basic RAG
- Multi‑document RAG
- Persistent RAG
- Metadata/filtering
- Retrieval evaluation
- Observability
- Security experiments (prompt injection, RAG poisoning, etc.)
- Fine‑tuning / LoRA / QLoRA
- Local agents
- Agentic RAG
- MCP integration
- Secure agentic RAG

*Note:* Labs may be merged or split based on experimental outcomes. Each lab teaches a mechanism; experiments test hypotheses and produce measurable evidence.