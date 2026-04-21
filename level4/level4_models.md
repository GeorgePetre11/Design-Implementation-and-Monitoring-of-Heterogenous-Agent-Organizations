# Level 4 — Model Assignments

| Agent | Model | Role in Level 4 |
|---|---|---|
| **Engagement Manager** | `qwen3.5:9b` | Decomposes the client question into workstreams and reviews every intermediate output before it moves forward. Can send work back with specific revision feedback. Acts as quality gatekeeper. Fits fully in VRAM (~5.1 GB Q4) for fast turnaround. |
| **Market Researcher** | `qwen3.5:35b-a3b` | `search_web()` + `read_document()`. 35B MoE with only 3B active params — 35B-tier knowledge at 3B-tier speed. Stricter source requirements: every claim must have a URL, must read at least 4 full pages. Uses `/no_think` mode to minimize hallucination during factual synthesis. If EM sends it back, does a second research pass targeting the gaps. |
| **Financial Analyst** | `gpt-oss:20b` | No tools — native chain-of-thought math. Must explicitly cite which Market Research data point each number comes from. If a number can't be traced to MR data, it must be marked `[ASSUMED]`. Uses `/think` mode for complex calculations and `/no_think` for simple data extraction. EM reviews this. |
| **Risk Analyst** | `qwen3.5:9b` | `search_web()` + `read_document()` + `assess_risk()`. Now also receives Financial Analyst output and can flag financial risks that contradict FA's analysis. Must cite sources for every risk. Uses `/think` mode for analytical depth while still fitting in VRAM, freeing resources for concurrent agents. |
| **Strategy Consultant** | `gemma4:31b` | No tools — synthesizes everything. Strict constraint: cannot introduce ANY claim not present in MR/FA/RA outputs. Top-tier writing and synthesis; GPU+RAM split (~19 GB Q4) is acceptable because it runs once per engagement. EM reviews the final report and can send it back if unsupported claims are found. |

Note: The Evaluator is a separate application and is not part of the Level 4 pipeline.
