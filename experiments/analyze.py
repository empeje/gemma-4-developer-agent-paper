"""Aggregate results/localize_*.jsonl into results/summary.json (+ markdown table). 95% bootstrap CIs over tasks."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
random.seed(0)
TAG = sys.argv[1] if len(sys.argv) > 1 else "e2b"
RES = ROOT / "results" / TAG

def ci(xs, B=2000):
    ms = sorted(sum(random.choices(xs, k=len(xs))) / len(xs) for _ in range(B))
    return sum(xs) / len(xs), ms[int(.025 * B)], ms[int(.975 * B)]

rows = {}
WANT = sys.argv[2:]
for p in sorted(RES.glob("localize_*.jsonl")):
    if WANT and p.stem[9:] not in WANT: continue
    rows[p.stem[9:]] = {r["instance_id"]: r for r in map(json.loads, open(p))}
common_ids = set.intersection(*[set(v) for v in rows.values()])

# non-LLM retrieval baselines (BM25 order, no model)
tasks = {t["instance_id"]: t for t in load_tasks()}
base = {"BM25_top1": [], "BM25_top3": []}
cache = {}
for i in sorted(common_ids):
    t = tasks[i]; g = rows[next(iter(rows))][i]["gold"]
    key = (t["repo"].split("/")[1], t["base_commit"])
    if key not in cache: cache[key] = Index(snapshot(*key))
    s = cache[key].search(t["problem_statement"][:6000], 3)
    base["BM25_top1"].append(int(s[0] in g)); base["BM25_top3"].append(int(any(x in g for x in s)))

out = {"n_tasks": len(common_ids), "systems": {}}
def add(name, hit1, hit3, extra=None):
    out["systems"][name] = {"hit@1": ci(hit1), "hit@3": ci(hit3), **(extra or {})}
for k, v in base.items():
    pass
add("BM25 (no LLM)", base["BM25_top1"], base["BM25_top3"])
for strat, d in rows.items():
    ids = sorted(common_ids)
    h1 = [int(bool(d[i]["pred"]) and d[i]["pred"][0] in d[i]["gold"]) for i in ids]
    h3 = [int(any(p in d[i]["gold"] for p in d[i]["pred"])) for i in ids]
    valid = [sum(d[i]["valid"]) / 3 for i in ids]
    ex = {"valid_path_rate": sum(valid) / len(valid)}
    if d[ids[0]]["n_cands"] is not None:
        ex["mean_cands"] = sum(d[i]["n_cands"] for i in ids) / len(ids)
        ex["cand_recall"] = sum(d[i]["cand_recall"] for i in ids) / len(ids)
    add(strat, h1, h3, ex)
# per-repo breakdown for hit@3
per = {}
for strat, d in rows.items():
    for repo in {d[i]["repo"] for i in common_ids}:
        ids = [i for i in common_ids if d[i]["repo"] == repo]
        per.setdefault(strat, {})[repo] = (sum(any(p in d[i]["gold"] for p in d[i]["pred"]) for i in ids) / len(ids), len(ids))
out["per_repo_hit@3"] = per
json.dump(out, open(RES / "summary.json", "w"), indent=1)
f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
print(f"n={out['n_tasks']}\n| system | hit@1 | hit@3 | valid | cands | cand-recall |\n|---|---|---|---|---|---|")
for k, v in out["systems"].items():
    print(f"| {k} | {f(v['hit@1'])} | {f(v['hit@3'])} | {v.get('valid_path_rate','-') if 'valid_path_rate' not in v else round(v['valid_path_rate'],2)} | {round(v['mean_cands'],1) if 'mean_cands' in v else '-'} | {round(v['cand_recall'],3) if 'cand_recall' in v else '-'} |")
print(json.dumps(per, indent=1))
