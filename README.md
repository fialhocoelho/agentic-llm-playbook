# agentic-llm-playbook
**agentic-llm-playbook** is a 4-week, build-first roadmap for modern LLM work—covering Transformers from scratch, training fundamentals, post-training (DPO/RLHF/GRPO), and agentic RAG with memory and evaluation. It’s in active development :rocket:

This repository is a 4-week, build-first playbook for modern LLM work — Transformers → post-training → agentic systems → evaluation.
It is in active development and currently focused on Week 1.

IMPORTANT: Create README.md with the EXACT content below, copied verbatim (do not rewrite, do not paraphrase, do not change tone, headings, or wording). Only ensure markdown formatting is preserved.

---BEGIN README.md---
# agentic-llm-playbook
A 4-week, build-first playbook for modern LLM work — **Transformers → post-training → agentic systems → evaluation**.

This repo is **in active development**. Right now I’m in **Week 1**.

The idea is to build a reusable “course-like” playbook with:
- short explanations (notes)
- runnable demos (notebooks + CLI)
- clean, minimal implementations
- tests for correctness
- weekly deliverables you can actually point to

---

## Status
- ✅ Week 1: **in progress**
- ⏳ Weeks 2–4: **planned**
- Expect this README to evolve as modules land.

---

## What this playbook will cover (full scope)
By the end of the 4 weeks, this repo will include hands-on modules for:

### 1) Transformer fundamentals (from scratch)
- Scaled dot-product attention
- Causal masking (why it matters)
- Multi-head attention (shape discipline)
- Transformer blocks (Pre-LN), MLP, residuals, LayerNorm
- A minimal GPT-style model (`GPTMini`)
- Tiny training sanity checks + generation

### 2) Training systems (tiny pretraining + engineering)
- Dataset pipeline choices (char-level vs token-level)
- Training loop (AdamW, warmup, gradient clipping)
- Checkpoint/resume, reproducibility, seeding
- Evaluation loop (loss/perplexity), sampling during training
- Debugging training failures + stability checks
- Simple experiment tracking (configs + results)

### 3) Post-training & preference optimization
- RLHF map: SFT → reward model → policy optimization (concept + minimal demo)
- Preference data: chosen vs rejected pairs
- **DPO** (core implementation + experiments)
- **GRPO** (conceptual deep dive + (optional) simplified implementation)
- Failure modes: reward hacking, KL collapse, dataset artifacts
- Before/after comparisons and small result tables

### 4) Agentic systems + RAG + memory
- Agent loop patterns (plan/execute, tool use, reflection)
- Retrieval basics (RAG) + long-context trade-offs
- Memory hooks (user preferences + interaction summaries)
- Personalization-oriented behaviors (context-aware responses)
- A small capstone: “agentic RAG with memory” end-to-end demo

### 5) Evaluation & “applied science” workflow
- Offline evaluation harness for LLM features
- Behavioral checks (consistency, instruction-following, style)
- Basic metrics (task success proxies) + simple scoring
- “LLM-as-a-judge” template (optional, reproducible)
- How to write results: clear tables, ablations, takeaways

### 6) Production-ish considerations (lightweight but real)
- Latency basics (KV-cache intuition, batching mindset)
- Reliability + guardrails mindset (tests, regressions)
- Clean code, modular design, runnable commands

---

## Current focus: Week 1
Week 1 is all about implementing the core blocks and building intuition.

**Week 1 deliverables (in progress)**
- Scaled dot-product attention + causal mask
- Multi-head attention
- Transformer block (Pre-LN)
- `GPTMini` forward + loss + simple generation
- Demos:
  - `notebooks/01_attention_playground.ipynb` (masked vs unmasked heatmaps)
  - `notebooks/02_gptmini_forward.ipynb` (shapes + loss + generate)
  - `notebooks/03_micro_train_overfit.ipynb` (tiny overfit sanity run)
- Tests for: masking correctness, shapes, gradient flow
- Notes: `notes/week01.md`

---

## Quickstart
```bash
make setup
make kernel
make test
