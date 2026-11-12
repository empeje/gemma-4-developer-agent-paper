"""Paired analyses on existing results: selection accuracy conditional on gold in candidate set; paired bootstrap diffs."""
import json, random, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT
random.seed(1)
TAG = sys.argv[1] if len(sys.argv) > 1 else "e2b"
RES = ROOT / "results" / TAG
S = sys.argv[2:] or ["A_nocontext","B_tree","C_bm25_20","D_graph","E_bm25_matched","F_graph_edges","G_callgraph","H_callgraph_edges","J_bm25_matched_call"]
R = {s: {r["instance_id"]: r for r in map(json.loads, open(RES / f"localize_{s}.jsonl"))} for s in S}
ids = sorted(set.intersection(*[set(v) for v in R.values()]))
h3 = lambda s, i: int(any(p in R[s][i]["gold"] for p in R[s][i]["pred"]))
h1 = lambda s, i: int(bool(R[s][i]["pred"]) and R[s][i]["pred"][0] in R[s][i]["gold"])
out = {"conditional_hit@3_given_gold_in_candidates": {}, "paired_diff_hit@3": {}, "paired_diff_hit@1": {}}
for s in S:
    sub = [i for i in ids if R[s][i]["cand_recall"] == 1]
    if sub: out["conditional_hit@3_given_gold_in_candidates"][s] = (sum(h3(s, i) for i in sub) / len(sub), len(sub))
def pdiff(a, b, f):
    d = [f(a, i) - f(b, i) for i in ids]
    rng = random.Random(zlib.crc32(f'{a}|{b}|{f.__name__}'.encode()))  # order-independent seed
    ms = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(5000))
    return sum(d) / len(d), ms[125], ms[4875]
for a, b in [("D_graph","E_bm25_matched"),("C_bm25_20","B_tree"),("E_bm25_matched","B_tree"),("D_graph","C_bm25_20"),("E_bm25_matched","C_bm25_20"),
               ("F_graph_edges","E_bm25_matched"),("F_graph_edges","D_graph"),
               ("G_callgraph","J_bm25_matched_call"),("H_callgraph_edges","J_bm25_matched_call"),("H_callgraph_edges","G_callgraph")]:
    if a in R and b in R:
        out["paired_diff_hit@3"][f"{a} - {b}"] = pdiff(a, b, h3)
        out["paired_diff_hit@1"][f"{a} - {b}"] = pdiff(a, b, h1)
# D-only vs E-only gold-in-candidates: where graph adds gold the BM25 control lacks
only_d = [i for i in ids if R["D_graph"][i]["cand_recall"] == 1 and R["E_bm25_matched"][i]["cand_recall"] == 0]
only_e = [i for i in ids if R["D_graph"][i]["cand_recall"] == 0 and R["E_bm25_matched"][i]["cand_recall"] == 1]
out["graph_unique_recall_tasks"] = {"graph_only": len(only_d), "bm25_only": len(only_e),
    "hit3_on_graph_only": sum(h3("D_graph", i) for i in only_d)}
json.dump(out, open(RES / "paired.json", "w"), indent=1)
print(json.dumps(out, indent=1))
