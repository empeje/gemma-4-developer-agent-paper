# A Laptop-Scale Recipe for Issue Localization: Precise Shortlists Let a 2B Gemma 4 Model Find the Right Files

*A controlled study of context construction for file-level localization with Gemma 4 on consumer hardware, no fine-tuning required*

## Abstract

Offline software-engineering agents must first decide which files to read, and for small local models the context window is the scarce resource. We show that a simple, fully local recipe works well: shortlist files with BM25, then let a small instruction-tuned model choose. On 125 real, post-cutoff issues from four Python repositories (the public tasks of the Gemma 4 Developer Agent competition), Gemma 4 E2B (4-bit, running on a laptop with no fine-tuning) places a gold file in its top three for **65.6%** of issues, a gain of **+12.8 points** over the BM25 ranking it was given (paired 95% CI [3.2, 22.4]) and **+16.0 points** over showing the model the entire file tree (CI [6.4, 25.6]). A controlled comparison shows the recipe does not need code-graph machinery: expanding the shortlist with import-graph neighbours raises candidate recall (0.88 to 0.95) but a size-matched BM25 control does equally well (hit@3 0.648 vs 0.624; difference -2.4 points, CI [-8.8, 4.0]), so the simplest pipeline is also the cheapest. Finally, a recall-versus-selection decomposition shows where the remaining headroom is: when the gold file is on screen, the model selects it 66-75% of the time, which identifies model selection, rather than retrieval, as the most promising target for tuning. Code and per-task predictions are released for exact reproduction.

## 1. Introduction

Agentic software-engineering systems such as SWE-agent and Agentless first localize (decide where to edit) and then repair. Putting this step on a laptop-sized model makes private, offline coding assistants practical, but the prompt budget forces a choice about what to show. A recurring idea, behind work such as RepoGraph, is that the *structure* of the codebase supplies relevant files that lexical retrieval misses.

We ask which ingredients of the context actually matter for a small local model, using a protocol that holds the number of candidate files fixed so that "more files" is not mistaken for "better structure":

> **RQ1.** Does a retrieved shortlist help a small model more than showing everything?
> **RQ2.** At equal candidate count, does import-graph expansion add value over plain lexical retrieval?
> **RQ3.** Where is the remaining headroom: candidate recall or model selection?

Contributions: (i) a laptop-scale recipe, BM25 shortlist followed by LLM selection, with a measured, statistically supported gain over both of its components; (ii) a size-matched protocol for evaluating structural context, with the finding that structure is not needed at this scale; (iii) a decomposition that locates the headroom in model selection; (iv) a reproducible harness (about 200 lines, about 11 minutes of inference for the 2B model on an Apple M1 Max).

## 2. Related Work

*Benchmarks and agents.* SWE-bench (Jimenez et al., 2024) evaluates resolution of real GitHub issues; SWE-agent (Yang et al., 2024) shows that the agent-computer interface matters, and Agentless (Xia et al., 2024) shows that a simple localize-then-repair pipeline is competitive. Our localization step mirrors Agentless's file-level stage but isolates the context-construction choice.
*Structural context.* RepoGraph (Ouyang et al., 2024) and repository maps used in coding assistants inject graph or symbol structure into the prompt, reporting gains over non-structural baselines. We differ by holding candidate count fixed.
*Context length effects.* Liu et al. (2024) show that language models use long contexts unreliably; this is consistent with our finding that more candidates do not help.
*Lexical retrieval.* BM25 (Robertson & Zaragoza, 2009) remains a strong baseline for code retrieval (Wang et al., 2024).
*Model.* Gemma 4 (Google DeepMind, 2026) provides E2B to 31B open-weight instruction-tuned variants; we use 4-bit MLX builds.

## 3. Method

**Data.** The 129 public tasks of the Gemma 4 Developer Agent competition (fastapi 67, rich 48, requests 13, httpx 1), each with issue text, base commit, and gold patch. They postdate typical training cutoffs. Localization target: the existing, non-test `.py` files modified by the gold patch (files created by the patch cannot be localized). 125 tasks have at least one such file (63 fastapi, 48 rich, 13 requests, 1 httpx); mean 1.73 gold files per task.

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

n = 125 tasks; 95% bootstrap CIs in brackets.

| System | hit@1 | hit@3 | valid paths | mean cands | cand. recall |
|---|---|---|---|---|---|
| BM25 alone (no LLM) | 0.328 [.248,.408] | 0.528 [.440,.616] | - | - | - |
| A: no context | 0.104 [.056,.160] | 0.144 [.088,.208] | 0.09 | - | - |
| B: full file tree | 0.376 [.288,.464] | 0.496 [.408,.584] | 0.98 | 77.2 | 1.00 |
| C: BM25 top-20 | 0.440 [.352,.528] | 0.656 [.568,.736] | 0.99 | 19.9 | 0.88 |
| D: BM25 top-5 + import graph | 0.480 [.392,.568] | 0.624 [.536,.704] | 0.98 | 38.9 | 0.95 |
| E: BM25 size-matched to D | 0.488 [.408,.576] | 0.648 [.568,.728] | 0.97 | 38.9 | 0.94 |

### 4.2 Replication with Gemma 4 12B

*The 12B replication (strategies C, E, D) was still running when this draft was rendered; its results are not yet included.*

### 4.3 Analysis

**Context is necessary.** Without candidates (A), 91% of predicted paths do not exist; hit@3 is 0.144.

**Narrow retrieval beats a full tree.** C improves on B by 16.0 points of hit@3 (paired 95% CI [6.4, 25.6]) although B's candidate recall is 1.0. Showing everything is worse than showing a good shortlist; a plausible (untested) explanation is that irrelevant files dilute the small model's attention. The LLM also improves on the lexical ranking it was given: +11.2 points of hit@1 (paired CI [1.6, 20.8]) and +12.8 points of hit@3 (CI [3.2, 22.4]) over BM25 alone (`results/e2b/llm_vs_bm25.json`).

**Graph expansion does not help once size is controlled.** D reaches higher candidate recall than C (0.95 vs 0.88) but a lower hit@3 (-3.2 points, CI [-11.2, 4.8]). Against the size-matched control E, the difference is -2.4 points (CI [-8.8, 4.0]) on hit@3 and -0.8 points (CI [-6.4, 4.8]) on hit@1. Graph neighbours supply the gold file that BM25 misses in only 4 tasks, while BM25 supplies one that the graph misses in 3. The graph's recall gain over C is mostly a size effect: E, with the same number of files but no structure, recovers 0.94.

**Selection is the bottleneck.** Conditional on the gold file being among the candidates, hit@3 is 0.746 for C, 0.655 for D and 0.686 for E. A quarter to a third of cases fail even when the answer is on the screen. Gains in candidate recall cannot exceed this ceiling, which explains why recall improvements did not translate into hits.

**By repository.** Hit@3 for C is 0.73 on fastapi (n=63), 0.56 on rich (n=48) and 0.62 on requests (n=13); the ordering of strategies is similar across repositories, though with these sample sizes per-repository differences are not significant.

## 5. Limitations

Four repositories (two dominate), 125 tasks: CIs are wide and small effects, such as a few points, cannot be excluded. File-level hits are a proxy for end-to-end resolution, which we did not measure. The import graph is one structure; call graphs or symbol-level context might behave differently. Greedy decoding with one prompt template. The 4-bit quantization may affect the models. The tasks come from a competition dataset, and we do not redistribute them.

## 6. Conclusion

A small Gemma 4 model on a laptop can localize issues usefully with nothing more than a lexical shortlist and a well-posed selection prompt, and that simple recipe beats both its retrieval component and a show-everything prompt. Because structural expansion brought no measurable benefit once candidate count was controlled, practitioners can skip graph construction for this step and spend the saved effort where our decomposition points: improving the model's selection among a handful of plausible files, for instance by fine-tuning or reranking, which is a natural next step for the Gemma 4 family.

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
