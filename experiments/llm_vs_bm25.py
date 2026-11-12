"""Paired bootstrap: LLM pipeline (strategy C) vs BM25 top-3 alone. Usage: python experiments/llm_vs_bm25.py [tag]"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
random.seed(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "e2b"
R = {r["instance_id"]: r for r in map(json.loads, open(ROOT / "results" / tag / "localize_C_bm25_20.jsonl"))}
T = {t["instance_id"]: t for t in load_tasks()}
cache, d1, d3 = {}, [], []
for i, r in R.items():
    t = T[i]; key = (t["repo"].split("/")[1], t["base_commit"])
    if key not in cache: cache[key] = Index(snapshot(*key))
    b = cache[key].search(t["problem_statement"][:6000], 3); g = r["gold"]
    d1.append(int(bool(r["pred"]) and r["pred"][0] in g) - int(b[0] in g))
    d3.append(int(any(p in g for p in r["pred"])) - int(any(x in g for x in b)))
out = {}
for n, d in [("hit@1", d1), ("hit@3", d3)]:
    ms = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(5000))
    out[n] = {"diff": round(sum(d) / len(d), 3), "ci95": [round(ms[125], 3), round(ms[4875], 3)]}
print(json.dumps(out))
