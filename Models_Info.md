# Model Selection — Multi-Agent Consulting System

The system uses **six distinct model families** across four complexity levels, creating genuine heterogeneity in model architecture, size, and provider. All pipeline agents run locally via Ollama; the Evaluator uses Gemini 2.5 Flash via Google AI Studio.

## Model Assignments by Level

### Level 1 — Single Agent (Baseline)

| Agent Role | Model | Size | Why This Model |
|---|---|---|---|
| **Single Agent** | qwen2.5:14b | 14B | Baseline model; produces entire report end-to-end including self-evaluation |

### Level 2 — Three Agents (Core Roles)

| Agent Role | Model | Size | Why This Model |
|---|---|---|---|
| **Engagement Manager** | qwen3:8b | 8B | Fast structured decomposition; thinking toggle; crisp instruction following |
| **Market Researcher** | qwen3:14b | 14B | 128K context; broad knowledge; excellent synthesis |
| **Strategy Consultant** | qwen3:32b | 32B | Superior business writing; nuanced argumentation; 95.2 on ArenaHard |

### Level 3 — Five Agents (Full Specialization)

| Agent Role | Model | Size | Why This Model |
|---|---|---|---|
| **Engagement Manager** | qwen3:8b | 8B | Fast structured decomposition; thinking toggle |
| **Market Researcher** | qwen3:14b | 14B | 128K context; broad knowledge; excellent synthesis |
| **Financial Analyst** | deepseek-r1:14b | 14B | MATH-500: 93.9%; purpose-built for quantitative reasoning; explicit chain-of-thought |
| **Risk Analyst** | qwen3:14b | 14B | Edge-case thinking; /think mode for analytical depth |
| **Strategy Consultant** | qwen3:32b | 32B | Superior business writing; nuanced argumentation |

### Level 4 — Six Agents (Hybrid Hierarchical)

| Agent Role | Model | Size | Why This Model |
|---|---|---|---|
| **Engagement Manager** | qwen3.5:9b | 9B | Fast decomposition + intermediate review; fits VRAM; uses /think mode toggles |
| **Market Researcher** | qwen3.5:35b-a3b | 35B (MoE, 3B active) | 35B-tier knowledge at 3B-tier speed; broad factual synthesis; /no_think for factual work |
| **Financial Analyst** | gpt-oss:20b | 20B | Strong quantitative reasoning; /think for complex calculations, /no_think for extraction |
| **Risk Analyst** | qwen3.5:9b | 9B | Analytical reasoning with /think mode; fits VRAM, freeing resources for concurrent agents |
| **Strategy Consultant** | gemma4:31b | 31B | Top-tier writing and synthesis; GPU+RAM split; no thinking-mode toggle |
| **Evaluator** | gemini-2.5-flash | — | Google AI Studio free tier (~10 RPM); strong analytical reasoning; OpenAI-compatible API |

---

## Why each model was chosen

**Engagement Manager (qwen3:8b → qwen3.5:9b)** needs structured thinking and task decomposition — breaking a business question like "Should we enter the Southeast Asian market?" into discrete workstreams. These agents don't need deep domain knowledge; they need crisp instruction following and structured output. The Qwen 3/3.5 small models with thinking mode enabled produce explicit `<think>` reasoning chains that naturally decompose problems into steps. At L4, the EM also reviews intermediate outputs from other agents, so the 9B model provides enough capability for quality assessment.

**Market Researcher (qwen3:14b → qwen3.5:35b-a3b)** synthesizes competitive intelligence, market trends, and customer segments. Needs broad factual knowledge, long-context processing for ingesting reports, and low hallucination rates. At L3, **Qwen 3 14B** supports 128K context natively and was trained on 18T tokens giving it deep knowledge across industries. At L4, the **Qwen 3.5 35B-A3B** MoE model provides 35B-tier knowledge with only 3B active parameters per token — meaning 35B-quality synthesis at the inference speed of a 3B model.

**Financial Analyst (deepseek-r1:14b → gpt-oss:20b)** is the role where a specialist model decisively beats the generalist. At L3, **DeepSeek R1 Distill 14B** scores 93.9% on MATH-500 — outperforming models four times its size on quantitative tasks. Its explicit chain-of-thought reasoning shows every calculation step, making it easy for the Evaluator to verify financial projections. At L4, **GPT-OSS 20B** provides even stronger quantitative reasoning with /think and /no_think mode toggles.

**Risk Analyst (qwen3:14b → qwen3.5:9b)** needs to think about edge cases and failure modes. At L3, the 14B model provides ample reasoning capacity. At L4, the smaller 9B model fits in VRAM alongside other models, and the /think mode provides sufficient analytical depth for risk identification.

**Strategy Consultant (qwen3:32b → gemma4:31b)** does the highest-value writing work — synthesizing findings into recommendations with options, tradeoffs, and persuasive argumentation. At L3, **Qwen 3 32B** scores 95.2 on ArenaHard and produces notably coherent long-form business writing. At L4, **Gemma 4 31B** (Google) provides top-tier writing and synthesis quality with a GPU+RAM split for memory efficiency.

**Evaluator (gemini-2.5-flash)** is the independent quality judge. Using a cloud model from a different provider (Google) than all pipeline agents (Ollama/local) maximizes heterogeneity. Gemini 2.5 Flash provides strong analytical reasoning for rubric-based evaluation, and the free tier is sufficient for thesis experiments. The OpenAI-compatible API means minimal code changes — just a base URL swap in the OpenAI Python SDK.

---

## Model diversity across levels

The progressive complexity experiment benefits from increasing model diversity at each level:

- **L1:** Single model (qwen2.5:14b) — 1 family, 1 provider
- **L2:** Qwen 3 family only (8B/14B/32B) — 1 family, 1 provider, 3 sizes
- **L3:** Qwen 3 + DeepSeek R1 — 2 families, 1 provider, 3 sizes
- **L4:** Qwen 3.5 + GPT-OSS + Gemma 4 + Gemini — 4 families, 2 providers (Ollama + Google AI Studio), 4 sizes

This progression from homogeneous to highly heterogeneous is a core thesis argument: more diverse model selection, matched to agent roles, produces better output quality.

---

## Ollama configuration

Set these environment variables for optimal multi-agent performance:

- `OLLAMA_FLASH_ATTENTION=1` — reduces memory usage, speeds up attention
- `OLLAMA_KV_CACHE_TYPE=q8_0` — halves KV cache memory with minimal quality impact
- `OLLAMA_KEEP_ALIVE=-1` — prevents model unloading between agent calls

When calling models via the Ollama API, use the `/api/chat` endpoint with `"options": {"num_ctx": 8192}` for small inputs (EM decomposition) and `"options": {"num_ctx": 32768}` for research loops, synthesis, reviews, and downstream agents that receive cumulative context.

### Pull commands

```bash
# Level 1
ollama pull qwen2.5:14b

# Level 2-3
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b
ollama pull deepseek-r1:14b   # L3 Financial Analyst

# Level 4
ollama pull qwen3.5:9b        # EM + RA
ollama pull qwen3.5:35b-a3b   # MR (MoE)
ollama pull gpt-oss:20b       # FA
ollama pull gemma4:31b         # SC
```

The Evaluator (gemini-2.5-flash) requires a Google AI Studio API key set via `EVALUATOR_API_KEY` or `GEMINI_API_KEY` environment variable.
