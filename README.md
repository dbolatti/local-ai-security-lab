# local-ai-security-lab

## Project Purpose
This repository is an experimental research and learning environment focused on studying, implementing, measuring, and security-testing local Artificial Intelligence systems.

## General Objective
To progressively build understanding of AI components from the ground up, avoiding frameworks that hide fundamental mechanisms, and to integrate cybersecurity considerations from the earliest stages.

## Experimental Philosophy
- **Bottom-up methodology**: Start with individual components (LLM, inference runtime, tokenizer, etc.) before assembling them.
- **Explicit component distinction**: Clearly separate responsibilities of LLM, inference runtime, tokenizer, embedding model, retriever, vector store, application/orchestration layer, RAG pipeline, and authoritative knowledge sources.
- **Security as a transversal concern**: Treat cybersecurity topics (model provenance, prompt injection, RAG poisoning, etc.) as integral to each phase.

## Initial Hardware Environment (Environment A)
- **OS**: Windows 11 host
- **CPU**: Intel Core i7-4510U (2 cores / 4 threads)
- **RAM**: ~16 GB
- **Graphics**: Intel integrated graphics (no NVIDIA CUDA GPU)
- **Execution mode**: CPU-only
- **Target models**: Initially ~1B-4B quantized models; larger quantized models may be tested experimentally later.

## Future Environment (Environment B)
- **OS**: Kali Linux virtual machine
- **Purpose**: Controlled security and adversarial testing.

## High-Level Architecture
The project will explore the following layers and their interactions:
1. **LLM** – The core language model that generates text given a prompt.
2. **Inference Runtime** – Software that executes the LLM (e.g., llama.cpp; candidates such as vLLM or TensorRT‑LLM are noted for future GPU‑capable environments but not relevant to the current CPU‑only baseline).
3. **Tokenizer** – Converts text to tokens and vice‑versa, specific to the LLM.
4. **Knowledge Sources** – Documents or databases that may be trusted, untrusted, or of varying provenance; they are not automatically trusted.
5. **Ingestion Pipeline** – Processes knowledge sources (e.g., loading, chunking, cleaning) before embedding.
6. **Embedding Model** – Produces vector representations of text chunks.
7. **Vector Store / Index** – Persists and indexes embeddings for similarity search (candidates: FAISS, Chroma, Qdrant).
8. **Retriever** – Searches the vector store for embeddings similar to a query embedding.
9. **Prompt Construction / Orchestration Layer** – Combines retrieved context with the user query into a prompt for the LLM (may include re‑ranking, filtering, etc.).
10. **RAG Pipeline** – The overall application architecture that couples retrieval (layers 5‑9) with generation (layer 1) to produce answers; it is not an intrinsic capability of the LLM itself but a composable system.

## Security Approach
From the first lab, we will examine:
- Model provenance and supply chain risks
- Trust boundaries between components
- Prompt injection (direct and indirect)
- Malicious document ingestion
- RAG poisoning and embedding attacks
- Unauthorized knowledge access and data leakage
- Metadata and ACL attacks
- Agent and MCP security

## Repository Organization
- `README.md` – This file.
- `ROADMAP.md` – Phased plan and lab progression.
- `LEARNING.md` – Experimental journal capturing findings, principles, decisions, and open questions.
- `docs/` – Detailed documentation split into:
  - `architecture/` – Diagrams and design docs.
  - `concepts/` – Explanations of AI and security concepts.
  - `security/` – Threat models, mitigations, and experimental results.
- `experiments/` – Structured experiments with hypotheses, variables, and measurable evidence.
- `labs/` – Hands‑on labs that teach mechanisms (distinct from experiments).
- `.gitignore` – Standard ignore patterns for Python/AI projects.

## Getting Started
No dependencies are installed yet. Future labs will guide the installation of specific runtimes and models.

---
*Documentation only – no code or dependencies installed at this stage.*