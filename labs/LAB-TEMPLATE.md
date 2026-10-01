# Lab Template

Every lab in `labs/` follows this structure. The goal of a lab is to **teach a mechanism**; the goal of an experiment (in `experiments/`) is to **test a hypothesis with measurable evidence**. Labs and experiments are distinct on purpose.

A lab must be executable later on the target machine (Environment A: Windows 11, Intel i7-4510U, 16 GB RAM, CPU-only). The design is theoretical until executed: never fabricate measurements or results.

## Status Banner

Every lab README starts with:

```
**Status:** THEORETICAL DESIGN — procedure defined, not yet executed.
```

## Required Sections (in this order)

### Title + Primary Question
`# LabNN – <Title>` followed by a **Primary question** in bold: the single mechanism this lab teaches, phrased as a question.

### Purpose
- Bullet list of the concepts this lab covers.
- **Topics reserved for later labs:** what is explicitly *not* explained here (avoids scope creep).
- **Explicitly stated:** which architectural components this lab does **NOT** yet contain (see the component list in the main `README.md`). Every lab must be honest about what is absent (e.g., no vector store before Lab08, no RAG before Lab11, no agents before Lab25).

### Environment Assumptions
Reference Environment A as *observed facts*, not hypotheses. State any additional assumptions.

### Candidate Technologies
Tools under consideration for this lab, framed as **candidates to be evaluated**, not decisions. Distinguish layers (e.g., Ollama is not the LLM; a runtime is not the model). Prefer minimal, mechanism-revealing tools; high-level frameworks stay postponed until the mechanism is understood.

### Prerequisites
- Which previous labs must be completed first (link them).
- What must be installed/available by this point (stated as a plan, not as done).

### Theoretical Background
The core of the lab. Precise definitions and the actual mechanics, including:
- How the mechanism works internally (data structures, algorithms, math where relevant).
- A **mermaid diagram** of the component architecture.
- Explicit formulas with the meaning of each variable (e.g., memory = parameters × bytes-per-weight; cosine similarity).
- Failure modes and trade-offs, not just the happy path.

### Hypothesis
`H<lab-number> (Working Hypothesis):` a falsifiable statement about behavior or performance that a later experiment will confirm or refute. Numbered to match the lab.

### Procedure (Planned Steps)
Ordered steps to execute later on the target machine. Commands may be proposed, but mark them as planned. Include what to observe at each step and what would count as surprising.

### Measurement / Success Criteria
- What must be recorded when the lab is executed, following the **Evidence Classification** in `LEARNING.md` (Observed, Measured Finding, etc.).
- A concrete completion checklist: how we know the lab objective was achieved.

### Security Analysis
Per-lab threat view, consistent with the project security approach:
- Trust boundaries introduced or crossed in this lab (label **TB1**, **TB2**, ... within this lab).
- Attack surface specific to this mechanism.
- Questions to consider (no validated mitigations yet — mitigations are evaluated in Phase 4 labs).

### Learning Objectives
Numbered checklist: "At the end of this lab, we should be able to explain: ..."

### Navigation
```
---
**Previous lab:** labs/lab<NN-1>-<name>/README.md
**Next lab:** labs/lab<NN+1>-<name>/README.md
*This document is part of the local-ai-security-lab repository.*
```

## Writing Rules

- **Language:** English, consistent with the existing repository.
- **Tone:** Precise, pedagogical, honest about uncertainty. Mark assumptions as hypotheses, facts as observed.
- **No fabricated results:** no invented benchmark numbers, outputs, or timings. Use "to be measured" placeholders.
- **Component isolation:** each lab studies its mechanism in isolation before integration, per the Architectural Principles in `LEARNING.md`.
- **Explicit interfaces:** when labs connect (e.g., chunks → embeddings → index), name the exact data passed between stages (text, token IDs, float vectors, metadata dicts).
- See `labs/lab01-local-llm/README.md` for the reference example of style and depth.
