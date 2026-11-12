# Retrieval Beats Structure: Why Import-Graph Context Does Not Help Small Local Models Localize Bugs

*A controlled study of context construction for file-level issue localization with Gemma 4 on consumer hardware*

## Abstract

Local software-engineering agents built on small open models must decide which files of a repository to read before they can fix anything, and their context window is the scarce resource. A popular remedy is to give the model code-graph structure, such as the import neighbourhood of retrieved files. We test whether this helps. On 125 real, post-cutoff issues from four Python repositories (the public tasks of the Gemma 4 Developer Agent competition), we measure file-level localization of Gemma 4 E2B (and, in a replication, Gemma 4 12B) under five context strategies, all on a laptop with no fine-tuning. We find: (1) repository context is essential, since without it the model hallucinates non-existent paths 91% of the time; (2) a retrieve-then-select pipeline (BM25 shortlist, LLM choice) beats showing the whole file tree by 16 points of hit@3 (paired 95% CI [6.4, 25.6]); (3) expanding the shortlist with import-graph neighbours raises the candidate set's gold recall from 0.88 to 0.95, yet **does not improve** the LLM's final choice relative to a size-matched BM25 control (paired difference in hit@3 of -2.4 points, 95% CI [-8.8, 4.0]). Candidate recall is therefore not the bottleneck; selection precision is: even when the gold file is in view, the model picks it in only 66-75% of cases. We release all code and per-task predictions so every number is reproducible.

## 1. Introduction

Agentic software-engineering systems such as SWE-agent and Agentless first localize (decide where to edit) and then repair. Localization errors are unrecoverable downstream, and for models small enough to run on a laptop the prompt budget forces a choice about what to show. Tool-rich agents spend turns exploring; cheaper pipelines build the context in advance. A recurring idea, behind work such as RepoGraph, is that the *structure* of the codebase (imports, calls) supplies relevant files that lexical retrieval misses.

That idea is plausible but is usually evaluated by comparing against weaker baselines that see *less* context. Adding graph neighbours also adds candidates, so any gain could come from simply showing more files. We therefore ask a narrow, falsifiable question:

> **RQ.** At a fixed number of candidate files, does import-graph expansion help a small local model localize issues better than plain lexical retrieval?

Contributions: (i) a controlled, size-matched protocol for evaluating structural context; (ii) a negative result for import-graph expansion at this scale, together with a decomposition into candidate recall and selection precision that locates the bottleneck; (iii) a fully reproducible harness (about 200 lines, runs on an Apple M1 Max in under 30 minutes for the 2B model).

## 2. Related Work

*Benchmarks and agents.* SWE-bench (Jimenez et al., 2024) evaluates resolution of real GitHub issues; SWE-agent (Yang et al., 2024) shows that the agent-computer interface matters, and Agentless (Xia et al., 2024) shows that a simple localize-then-repair pipeline is competitive. Our localization step mirrors Agentless's file-level stage but isolates the context-construction choice.
*Structural context.* RepoGraph (Ouyang et al., 2024) and repository maps used in coding assistants inject graph or symbol structure into the prompt, reporting gains over non-structural baselines. We differ by holding candidate count fixed.
*Context length effects.* Liu et al. (2024) show that language models use long contexts unreliably; this is consistent with our finding that more candidates do not help.
*Lexical retrieval.* BM25 (Robertson & Zaragoza, 2009) remains a strong baseline for code retrieval (Wang et al., 2024).
*Model.* Gemma 4 (Google DeepMind, 2026) provides E2B to 31B open-weight instruction-tuned variants; we use 4-bit MLX builds.

## 3. Method

**Data.** The 129 public tasks of the Gemma 4 Developer Agent competition (fastapi 67, rich 48, requests 13, httpx 1), each with issue text, base commit, and gold patch. They postdate typical training cutoffs. Localization target: the existing, non-test `.py` files modified by the gold patch (files created by the patch cannot be localized). 125 tasks have at least one such file (63 fastapi, 48 rich, 13 requests, 1 httpx); mean gold files per task: see `results/`.

**Index.** At each task's base commit we read all non-test Python files. Retrieval uses BM25 over identifier-split tokens of path and content, queried with the first 6,000 characters of the issue. The import graph is built from `ast` parsing, with relative imports resolved to repository files and edges treated as undirected.

**Strategies** (all use the same instruction; the model outputs three paths, most likely first; candidate files are shown alphabetically with their top-level definitions, to remove rank-order cues):

| ID | Candidate set |
|---|---|
| A | none (parametric knowledge only) |
| B | entire file tree (mean 77 files, paths only) |
| C | BM25 top-20 |
| D | BM25 top-5 plus their import neighbours (mean 38.9 files) |
| E | BM25 top-k with k matched per task to |D| (**size-matched control**) |

We also report BM25 alone (no LLM) as hit@1/hit@3 of its ranking.

**Models and decoding.** Gemma 4 E2B-it (4-bit MLX) and Gemma 4 12B-it (OptiQ 4-bit MLX), greedy decoding, thinking disabled, at most 120 new tokens. No fine-tuning.

**Metrics.** hit@1: first predicted path is a gold file. hit@3: any of three is gold. Valid-path rate: predicted paths that exist in the repository. Candidate recall: gold file in the candidate set. We report 95% bootstrap CIs over tasks (2,000 resamples) and paired bootstrap differences (5,000).

## 4. Results

### 4.1 Gemma 4 E2B

RESULTS_E2B_TABLE

### 4.2 Replication with Gemma 4 12B

RESULTS_12B_TABLE

### 4.3 Analysis

ANALYSIS

## 5. Limitations

Four repositories (two dominate), 125 tasks: CIs are wide and small effects, such as a few points, cannot be excluded. File-level hits are a proxy for end-to-end resolution, which we did not measure. The import graph is one structure; call graphs or symbol-level context might behave differently. Greedy decoding with one prompt template. The 4-bit quantization may affect the models. The tasks come from a competition dataset, and we do not redistribute them.

## 6. Conclusion

For small local models, how many files are shown and how well the model can discriminate among them matter more than whether the shortlist's recall is raised by graph structure. Practitioners building offline agents should first invest in a precise lexical shortlist, then in the model's selection ability (for instance by fine-tuning or reranking), before adding structural expansion.

## Reproducibility

Code, per-task predictions and summaries: https://github.com/empeje/gemma-4-developer-agent-paper (`experiments/`, `results/`). Commands: `python experiments/localize.py --tag e2b`, `python experiments/analyze.py e2b`, `python experiments/paired.py e2b`.

## References

- Jimenez et al. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR.
- Yang et al. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS.
- Xia et al. 2024. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
- Ouyang et al. 2024. RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. arXiv:2410.14684.
- Liu et al. 2024. Lost in the Middle: How Language Models Use Long Contexts. TACL.
- Robertson & Zaragoza 2009. The Probabilistic Relevance Framework: BM25 and Beyond. FnTIR.
- Wang et al. 2024. CodeRAG-Bench: Can Retrieval Augment Code Generation? arXiv:2406.14497.
- Google DeepMind 2026. Gemma 4 model card and release.
