# Learning Journal

## Purpose
This file captures experimental findings, architectural principles, decisions made, and open questions as we progress through the labs and experiments. It serves as a running record of what we have learned and what remains to be investigated.

## Initial Environment Findings (Observed)
- Windows 11 host
- Intel Core i7-4510U (2 cores / 4 threads)
- Approximately 16 GB RAM
- Intel integrated graphics
- No NVIDIA CUDA GPU detected
- CPU execution will be the initial baseline

## Working Hypothesis
- The hardware may be capable of running quantized models in the 1B‑4B parameter range at usable speeds for experimentation. This remains to be validated experimentally.

## Initial Architectural Principles
1. **Component Isolation** – Each AI component (LLM, runtime, tokenizer, embedding model, retriever, vector store, orchestration layer) will be studied and tested in isolation before integration.
2. **Explicit Interfaces** – Interactions between components will be via well‑defined, minimal interfaces (e.g., token IDs, embeddings, text strings) to facilitate substitution and measurement.
3. **Security‑First Mindset** – Threat modeling and mitigations are considered from the first lab; security is not an afterthought.
4. **Reproducibility** – Experiments will record hardware, model, quantization, context size, parameters, resource usage (RAM, CPU, VRAM when applicable), latency, tokens/second, inputs, outputs, and observations.
5. **Bottom‑Up Build‑Up** – Avoid high‑level frameworks that hide internal mechanics until we have understood the underlying mechanisms.

## Candidate Technologies (to be evaluated)
- **Inference Runtime**: Ollama (candidate for quick high‑level local inference), llama.cpp (candidate for lower‑level study of GGUF, quantization, CPU execution)
- **Tokenizer**: Derived from model’s own tokenizer (HuggingFace tokenizers) – to be evaluated per model
- **Embedding Model**: Sentence‑Transformers‑style models (candidates for conversion to GGML/ONNX) – to be evaluated
- **Vector Store**: FAISS (candidate for similarity search experiments), Chroma and Qdrant (candidates for later persistent vector‑store experiments)
- **Orchestration Layer**: Minimal Python scripts (initial candidate); LangChain/LlamaIndex intentionally postponed until underlying mechanisms are understood
- **Security Testing Environment**: Kali Linux VM (Environment B) – candidate for adversarial testing once basic pipeline is functional

## Open Questions
- What is the optimal quantization level (bits) for balancing size, speed, and accuracy on this CPU?
- How does chunk size affect retrieval quality and latency in a CPU‑only setting?
- Which FAISS index type provides the best trade‑off between memory footprint and query speed for our expected corpus sizes?
- What are the most effective mitigations against prompt injection when the LLM is served via a local CPU runtime?
- How can we measure and detect RAG poisoning in a controlled lab environment?
- What lightweight agent frameworks (if any) can be built on top of our component stack without obscuring underlying mechanisms?
- How does the Model Context Protocol (MCP) integrate with a locally hosted LLM and vector store?

## Evidence Classification
To maintain rigorous documentation, every entry in this journal will be labeled with one of the following categories:
- **Observed**: Directly measured or perceived facts (e.g., hardware specs, runtime behavior).
- **Decision**: A choice made by the project team based on available information.
- **Working Hypothesis**: A tentative assumption that requires experimental validation.
- **Candidate Technology**: A tool, library, or approach under consideration but not yet selected.
- **Measured Finding**: Result of a controlled experiment that includes quantified variables (hardware, model, quantization, context size, parameters, RAM/CPU/VRAM usage, latency, tokens/second, inputs, outputs, observations).