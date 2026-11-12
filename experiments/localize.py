"""File-level issue localization with Gemma 4 E2B under different context strategies.
Usage: python experiments/localize.py [--limit N] [--strategies A,B,...]
Writes results/localize_<strategy>.jsonl (one line per task) - every paper number derives from these."""
import argparse, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = str(ROOT / "models" / "gemma-4-e2b-it-4bit")
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

INSTR = ("You are a software engineer triaging a bug report. Decide which source files of the repository "
         "must be edited to resolve the issue. Answer ONLY with a JSON list of exactly 3 file paths, "
         "most likely first, e.g. [\"pkg/a.py\", \"pkg/b.py\", \"pkg/c.py\"].")

def build_prompt(strategy, task, ix):
    issue = task["problem_statement"][:4000]
    q = task["problem_statement"][:6000]
    head = f"Repository: {task['repo']}\n\nIssue:\n{issue}\n\n"
    seeds = ix.search(q, 40)
    if strategy == "A_nocontext":
        body = "Candidate files are not given; answer from your knowledge of the repository.\n"
        cands = None
    elif strategy == "B_tree":
        cands = ix.paths
    elif strategy == "C_bm25_20":
        cands = seeds[:20]
    elif strategy == "D_graph":
        s = set(seeds[:5])
        for p in seeds[:5]: s |= ix.graph[p]
        cands = sorted(s)
    elif strategy == "E_bm25_matched":   # size-matched control for D
        n = len(set(seeds[:5]).union(*[ix.graph[p] for p in seeds[:5]]))
        cands = ix.search(q, n)
    else: raise ValueError(strategy)
    if cands is not None:
        cards = [ix.summary(p) if strategy != "B_tree" else p for p in sorted(cands)]
        body = "Candidate files:\n" + "\n".join(cards) + "\n"
    return INSTR, head + body + "\nJSON list of 3 file paths:", cands

def parse(out, valid=None):
    m = re.search(r"\[[^\]]*\]", out, re.S)
    try: L = [x for x in json.loads(m.group(0)) if isinstance(x, str)]
    except Exception: L = re.findall(r"[\w./-]+\.py", out)
    return L[:3]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strategies", default="A_nocontext,B_tree,C_bm25_20,D_graph,E_bm25_matched")
    a = ap.parse_args()
    model, tokenizer = load(MODEL)
    tasks = [(t, gold_files(t)) for t in load_tasks()]
    tasks = [(t, g) for t, g in tasks if g]
    if a.limit: tasks = tasks[:a.limit]
    idx = {}
    for strat in a.strategies.split(","):
        out_path = RES / f"localize_{strat}.jsonl"
        done = {json.loads(l)["instance_id"] for l in open(out_path)} if out_path.exists() else set()
        f = open(out_path, "a"); t0 = time.time()
        for t, g in tasks:
            if t["instance_id"] in done: continue
            key = (t["repo"].split("/")[1], t["base_commit"])
            if key not in idx: idx[key] = Index(snapshot(*key))
            ix = idx[key]
            sysm, user, cands = build_prompt(strat, t, ix)
            msgs = [{"role": "user", "content": sysm + "\n\n" + user}]
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            out = generate(model, tokenizer, prompt=prompt, max_tokens=120, sampler=make_sampler(temp=0.0), verbose=False)
            pred = parse(out)
            rec = dict(instance_id=t["instance_id"], repo=t["repo"], strategy=strat, gold=g, pred=pred,
                       n_cands=None if cands is None else len(cands),
                       cand_recall=None if cands is None else int(any(x in cands for x in g)),
                       valid=[p in ix.files for p in pred], raw=out[:300])
            f.write(json.dumps(rec) + "\n"); f.flush()
        print(strat, "done", round(time.time() - t0), "s", flush=True)
