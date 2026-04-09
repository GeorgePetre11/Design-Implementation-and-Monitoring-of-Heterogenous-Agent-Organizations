# Level 4 — Model Assignments

| Agent | Model | Role in Level 4 |
|---|---|---|
| **Engagement Manager** | `qwen3:8b` | Decomposes the client question into workstreams and reviews every intermediate output before it moves forward. Can send work back with specific revision feedback. Acts as quality gatekeeper. |
| **Market Researcher** | `qwen3:14b` | `search_web()` + `read_document()`. Stricter source requirements: every claim must have a URL, must read at least 4 full pages. Uses `/no_think` mode to minimize hallucination during factual synthesis. If EM sends it back, does a second research pass targeting the gaps. |
| **Financial Analyst** | `qwen3:14b` | No tools — native chain-of-thought math. Must explicitly cite which Market Research data point each number comes from. If a number can't be traced to MR data, it must be marked `[ASSUMED]`. Uses `/think` mode for complex calculations and `/no_think` for simple data extraction. EM reviews this. |
| **Risk Analyst** | `qwen3:14b` | `search_web()` + `read_document()` + `assess_risk()`. Now also receives Financial Analyst output and can flag financial risks that contradict FA's analysis. Must cite sources for every risk. Uses `/think` mode for analytical reasoning. |
| **Strategy Consultant** | `qwen3.5:27b` | No tools — synthesizes everything. Strict constraint: cannot introduce ANY claim not present in MR/FA/RA outputs. 256K context window holds all intermediate outputs comfortably. EM reviews the final report and can send it back if unsupported claims are found. |
| **Evaluator** | Claude Opus 4.6 (API) | Independent final judge. Scores the final report on the 6-criterion rubric. Cannot modify the report. |
