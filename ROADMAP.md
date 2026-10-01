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

## Lab Index (Working Roadmap)

All labs are defined under `labs/` and follow `labs/LAB-TEMPLATE.md`. Each lab teaches a mechanism; experiments (under `experiments/`) test hypotheses and produce measurable evidence.

### Phase 1 – Local AI
- **Lab01** – [Running a Local LLM](labs/lab01-local-llm/README.md)
- **Lab02** – [Understanding the Inference Runtime](labs/lab02-inference-runtime/README.md)
- **Lab03** – [Tokens, Context Windows and Generation Parameters](labs/lab03-tokens-context-generation/README.md)
- **Lab04** – [Model Size and Quantization](labs/lab04-model-size-quantization/README.md)
- **Lab05** – [Measuring Local AI Performance](labs/lab05-measuring-performance/README.md)

### Phase 2 – Retrieval
- **Lab06** – [Embeddings and Similarity](labs/lab06-embeddings-similarity/README.md)
- **Lab07** – [Chunking Strategies](labs/lab07-chunking-strategies/README.md)
- **Lab08** – [FAISS Vector Store](labs/lab08-faiss-vector-store/README.md)
- **Lab09** – [Persistent Vector Store](labs/lab09-persistent-vector-store/README.md)
- **Lab10** – [Retrieval Pipeline](labs/lab10-retrieval-pipeline/README.md)

### Phase 3 – RAG
- **Lab11** – [Simple RAG](labs/lab11-simple-rag/README.md)
- **Lab12** – [Multi-Document RAG](labs/lab12-multi-document-rag/README.md)
- **Lab13** – [Persistent RAG](labs/lab13-persistent-rag/README.md)
- **Lab14** – [Metadata and Filtering](labs/lab14-metadata-filtering/README.md)
- **Lab15** – [Retrieval Evaluation](labs/lab15-retrieval-evaluation/README.md)
- **Lab16** – [RAG Observability](labs/lab16-rag-observability/README.md)

### Phase 4 – RAG and AI Security
- **Lab17** – [Direct Prompt Injection](labs/lab17-prompt-injection/README.md)
- **Lab18** – [Indirect Prompt Injection and Malicious Documents](labs/lab18-indirect-injection-malicious-docs/README.md)
- **Lab19** – [RAG Poisoning](labs/lab19-rag-poisoning/README.md)
- **Lab20** – [Embedding Attacks](labs/lab20-embedding-attacks/README.md)
- **Lab21** – [Data Leakage and Access Control](labs/lab21-data-leakage-access-control/README.md)
- **Lab22** – [Model Provenance and Supply Chain](labs/lab22-model-provenance-supply-chain/README.md)

### Phase 5 – Model Adaptation
- **Lab23** – [Fine-Tuning Fundamentals](labs/lab23-fine-tuning-basics/README.md)
- **Lab24** – [LoRA and QLoRA](labs/lab24-lora-qlora/README.md)

### Phase 6 – Agentic AI and MCP
- **Lab25** – [Local Agents and Tool Use](labs/lab25-local-agents/README.md)
- **Lab26** – [MCP Integration](labs/lab26-mcp-integration/README.md)
- **Lab27** – [Secure Agentic RAG (Capstone)](labs/lab27-secure-agentic-rag/README.md)

*Note:* Labs may be merged or split based on experimental outcomes. Each lab teaches a mechanism; experiments test hypotheses and produce measurable evidence.