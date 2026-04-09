# Optimal Ollama models for a multi-agent consulting system

**Qwen 3 is the dominant model family for this setup, but the real unlock is pairing it with DeepSeek R1 distills for quantitative roles and running different model sizes on each machine.** Your MacBook Pro M3 (16GB) can reliably handle 7–8B models at full GPU speed, while the Ryzen 7 (48GB) comfortably runs 32B models — creating a natural two-tier architecture where fast, lightweight agents live on the Mac and heavy reasoning agents live on the PC. Your current Qwen 2.5 14B baseline is a solid choice but likely runs poorly on the MacBook due to memory pressure; upgrading to Qwen 3 across the board yields significant gains at every parameter count.

The table below maps each agent role to its recommended model, then the sections that follow explain why.

| Agent Role | Recommended Model | Size | Machine | RAM Used | Why This Model |
|---|---|---|---|---|---|
| **L1 — Single Agent** | Qwen 3 14B Q4_K_M | 14B | Ryzen 7 | ~11 GB | Matches Qwen 2.5 32B; thinking toggle; strong writing + reasoning |
| **L2 — Consultant** | Qwen 3 32B Q4_K_M | 32B | Ryzen 7 | ~22 GB | Best dense all-rounder under 70B; handles research, analysis, and writing |
| **L3 — Engagement Mgr** | Qwen 3 8B (thinking ON) | 8B | MacBook M3 | ~6 GB | Fast structured decomposition; explicit chain-of-thought; 35–55 tok/s |
| **L3 — Market Researcher** | Qwen 3 14B Q4_K_M | 14B | Ryzen 7 | ~11 GB | 128K context; broad knowledge; excellent synthesis |
| **L3 — Strategy Consultant** | Qwen 3 32B Q4_K_M | 32B | Ryzen 7 | ~22 GB | Superior business writing; nuanced argumentation |
| **L4 — Engagement Mgr** | Qwen 3 4B or Phi-4 Mini Reasoning | 3.8–4B | MacBook M3 | ~3.3 GB | Ultra-fast task routing; structured output; frees Mac memory |
| **L4 — Market Researcher** | Qwen 3 14B Q4_K_M | 14B | Ryzen 7 | ~11 GB | Strong factual synthesis; multilingual for global markets |
| **L4 — Financial Analyst** | DeepSeek R1 Distill 14B Q4_K_M | 14B | Ryzen 7 | ~11 GB | MATH-500: 93.9%; purpose-built for quantitative reasoning |
| **Evaluator** | Claude Opus 4.6 (API) | — | Cloud | — | Already chosen; excellent for evaluation and feedback |

---

## Why Qwen 2.5 14B should be upgraded

Your current Level 1 baseline, Qwen 2.5 14B, was an excellent choice six months ago but has been decisively surpassed. **Qwen 3 14B matches or exceeds Qwen 2.5 32B** on reasoning benchmarks (AIME '24: 73.8% vs ~65%) while consuming the same memory. It adds a critical feature for multi-agent work: a **thinking/non-thinking toggle** that lets agents switch between deep chain-of-thought reasoning and fast direct responses per turn — no model swap needed.

On your MacBook M3 with 16GB, the 14B Q4_K_M model needs ~10.7GB at runtime, which exceeds the Mac's ~10.5GB GPU allocation. This means partial CPU offload and a speed drop from 35+ tok/s to roughly 10–20 tok/s. That's workable but frustrating. **For the MacBook, 8B models are the sweet spot** — they load entirely into GPU memory and generate at 35–55 tok/s. Reserve the 14B and 32B models for the Ryzen 7 PC, where 48GB of RAM gives comfortable headroom.

If you want to keep running a 14B on the Mac, use Q3_K_M quantization (~9GB) and cap context at 4096 tokens. Enable flash attention (`OLLAMA_FLASH_ATTENTION=1`) and KV cache quantization (`OLLAMA_KV_CACHE_TYPE=q8_0`) to claw back 1–2GB. But the honest recommendation: run Qwen 3 8B on the Mac and Qwen 3 32B on the PC.

---

## How each agent role maps to model strengths

**Engagement Managers (L3 and L4)** need structured thinking and task decomposition — breaking a business question like "Should we enter the Southeast Asian market?" into discrete workstreams (market sizing, regulatory analysis, competitive landscape, financial modeling). These agents don't need deep domain knowledge; they need crisp instruction following and structured output. Qwen 3 8B with thinking mode enabled produces explicit `<think>` reasoning chains that naturally decompose problems into steps. For the L4 Engagement Manager, which handles even more routine orchestration, **Phi-4 Mini Reasoning at 3.8B** is remarkably capable — it was RL-trained specifically for step-by-step reasoning and runs at 60+ tok/s on the M3, freeing memory for other processes.

**Market Researchers (L3 and L4)** synthesize competitive intelligence, market trends, and customer segments. They need broad factual knowledge, long-context processing for ingesting reports, and low hallucination rates. **Qwen 3 14B** excels here: it supports **128K token context** natively (critical for processing lengthy market reports), covers 29+ languages for international research, and was trained on 18 trillion tokens giving it deep knowledge across industries. Running on the Ryzen 7 at ~11GB, it leaves ample room for a second model to run concurrently.

**The Strategy Consultant (L3)** does the highest-value writing work — synthesizing findings into recommendations with options, tradeoffs, and persuasive argumentation. This demands the most capable model you can run. **Qwen 3 32B** is the clear winner: it scores 95.2 on ArenaHard (a writing-quality benchmark), surpasses DeepSeek R1 Distill 32B on general tasks while matching it on reasoning, and produces notably coherent long-form business writing. At ~22GB Q4_K_M, it's the largest dense model that runs comfortably on 48GB RAM. Using non-thinking mode for the actual writing and thinking mode for analysis gives you two capabilities in one model.

**The Financial Analyst (L4)** is the one role where a specialist model decisively beats the generalist. DeepSeek R1 Distill Qwen 14B scores **93.9% on MATH-500** and 69.7% on AIME — outperforming models four times its size on quantitative tasks. Its explicit chain-of-thought reasoning shows every calculation step, making it easy for the Evaluator (Claude Opus 4.6) to verify financial projections and catch errors. The R1 distill models were specifically trained via knowledge distillation from the full DeepSeek R1 671B on mathematical and logical reasoning tasks, giving them disproportionate quantitative strength for their parameter count.

---

## Practical deployment across two machines

The architecture should route agents to machines based on model size, not agent type. Here's how to think about it:

**MacBook M3 16GB — the "fast lane."** This machine handles Engagement Manager agents and any routing/classification tasks. Keep one model loaded at all times (`OLLAMA_KEEP_ALIVE=-1`). Set `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1`. The 8B model generates at 35–55 tok/s, making it feel responsive for decomposition tasks that produce shorter outputs. One important constraint: **standardize context length across all requests** (e.g., always use `num_ctx: 8192`) because Ollama treats different context sizes as different model configurations and will trigger a slow reload.

**Ryzen 7 48GB — the "heavy lifter."** This machine runs Consultant, Researcher, Strategy, and Financial Analyst agents. The critical question is whether this PC has a dedicated GPU. With **CPU-only inference**, expect roughly **2–5 tok/s for 32B models** and 5–10 tok/s for 14B models. That's slow but functional for generating consulting reports where latency tolerance is minutes, not seconds. With a **dedicated GPU** (e.g., RTX 3060 12GB or better), the 14B model runs at 30–40 tok/s and the 32B at 15–30 tok/s depending on VRAM — a transformative difference. If you're investing in this system, a 16–24GB GPU for the Ryzen is the single highest-impact upgrade.

With 48GB RAM, you can keep **two models loaded simultaneously**: the 32B Qwen 3 (~22GB) plus the 14B DeepSeek R1 Distill (~11GB) = ~33GB, leaving 15GB for the OS and applications. This means the Strategy Consultant and Financial Analyst agents can alternate without model swap delays. Configure this with `OLLAMA_MAX_LOADED_MODELS=2`.

For cross-machine routing, use **Olla** (a lightweight Go proxy at github.com/thushan/olla) to present a unified API endpoint. It supports intelligent model-aware routing — requests for `qwen3:8b` go to the MacBook, requests for `qwen3:32b` go to the Ryzen. Set `OLLAMA_HOST=0.0.0.0` on both machines to accept remote connections.

---

## The MoE wildcard worth testing

One model deserves special mention: **Qwen 3 30B-A3B**, a Mixture-of-Experts model with 30B total parameters but only **3B active per token**. It matches QwQ-32B (a dedicated 32B reasoning model) on math benchmarks while using dramatically less compute. At ~18GB Q4_K_M it fits on the Ryzen 7, but because only 3B parameters activate per token, inference speed is closer to a 3B model — potentially **15–25 tok/s even on CPU**. This makes it a compelling alternative for the Market Researcher role where you want broad knowledge (30B parameter knowledge base) with fast generation. The tradeoff is that MoE models can be less consistent than dense models on nuanced writing tasks, so test it against the dense Qwen 3 14B on your specific consulting prompts before committing.

---

## Recommended Ollama commands and configuration

To get started, pull these models on each machine:

On the **MacBook M3**: `ollama pull qwen3:8b` (Engagement Manager) and optionally `ollama pull phi4-mini-reasoning` (L4 EM). On the **Ryzen 7 PC**: `ollama pull qwen3:32b` (Consultant and Strategy), `ollama pull qwen3:14b` (Market Researcher), and `ollama pull deepseek-r1:14b` (Financial Analyst).

Set these environment variables on both machines for optimal multi-agent performance:

- `OLLAMA_FLASH_ATTENTION=1` — reduces memory usage, speeds up attention
- `OLLAMA_KV_CACHE_TYPE=q8_0` — halves KV cache memory with minimal quality impact
- `OLLAMA_KEEP_ALIVE=-1` — prevents model unloading between agent calls
- `OLLAMA_HOST=0.0.0.0` — enables cross-machine access

When calling models via the Ollama API, use the `/api/chat` endpoint with `"options": {"num_ctx": 8192}` consistently. For the Strategy Consultant and Market Researcher agents that need to process long documents, you can push to `num_ctx: 32768` on the Ryzen 7 — but be aware this adds ~4–8GB to memory usage for 32B models.

## Conclusion

The optimal architecture uses **four distinct models across two machines**: Qwen 3 8B on the MacBook for fast orchestration, and Qwen 3 32B, Qwen 3 14B, and DeepSeek R1 14B on the Ryzen 7 for heavy reasoning, research, and quantitative analysis. This replaces your Qwen 2.5 14B baseline with a generation-newer model family that offers thinking mode toggles, better reasoning, and stronger writing — while properly respecting your MacBook's 16GB memory ceiling. The single most impactful hardware upgrade would be adding a dedicated GPU to the Ryzen 7 PC, which would boost generation speed by 5–10× on the models that matter most. And the single most impactful model choice is using DeepSeek R1 Distill for the Financial Analyst — its specialist reasoning training makes it dramatically better at quantitative tasks than any generalist model at the same size.