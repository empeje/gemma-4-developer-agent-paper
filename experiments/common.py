"""Repo access, BM25 retrieval and import-graph utilities (no model dependency)."""
import ast, json, re, subprocess, functools
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def load_tasks():
    return [json.loads(l) for l in open(DATA / "tasks.jsonl")]

def git(repo, *args):
    return subprocess.run(["git", "-C", str(DATA / "repos" / f"{repo}.git"), *args],
                          capture_output=True, text=True, errors="replace").stdout

def is_test(p):
    return bool(re.search(r"(^|/)(tests?|docs_src|docs|scripts)(/|$)|(^|/)test_|conftest", p))

def gold_files(task):
    """Existing, non-test source files modified by the gold patch (new files are unlocalizable)."""
    n = task["repo"].split("/")[1]
    tracked = set(git(n, "ls-tree", "-r", "--name-only", task["base_commit"]).split("\n"))
    fs = set(re.findall(r"^--- a/(\S+)", task["patch"], re.M))
    return sorted(f for f in fs if f in tracked and f.endswith(".py") and not is_test(f))

@functools.lru_cache(maxsize=None)
def snapshot(repo, commit):
    """{path: source} for non-test .py files at commit."""
    out = {}
    for p in git(repo, "ls-tree", "-r", "--name-only", commit).split("\n"):
        if p.endswith(".py") and not is_test(p):
            out[p] = git(repo, "show", f"{commit}:{p}")
    return out

def tok(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [w for w in re.findall(r"[a-zA-Z]{2,}", text.lower())]

class Index:
    def __init__(self, files):
        self.paths = sorted(files)
        self.files = files
        self.bm25 = BM25Okapi([tok(p.replace("/", " ") + " " + p.replace("/", " ") + " " + files[p]) for p in self.paths])
        self.graph = self._graph()

    def search(self, query, k=20):
        sc = self.bm25.get_scores(tok(query))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        return [self.paths[i] for i in order]

    def _graph(self):
        """Undirected import graph between repo files."""
        mod2path = {}
        for p in self.paths:
            m = p[:-3].replace("/", ".")
            if m.endswith(".__init__"): m = m[:-9]
            mod2path[m] = p
            mod2path[m.split(".", 1)[-1] if "." in m else m] = mod2path.get(m, p)
        g = {p: set() for p in self.paths}
        for p, src in self.files.items():
            try: tree = ast.parse(src)
            except SyntaxError: continue
            pkg = p[:-3].split("/")[:-1]
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = ".".join(pkg[:len(pkg) - node.level + 1]) if node.level else ""
                    name = ((base + ".") if base and node.module else base) + (node.module or "")
                    mods = [name] + [name + "." + a.name for a in node.names]
                for m in mods:
                    q = mod2path.get(m)
                    if q and q != p:
                        g[p].add(q); g[q].add(p)
        return g

    def summary(self, p, n=6):
        """Compact file card: path + top-level defs."""
        defs = re.findall(r"^(?:class|def|async def)\s+(\w+)", self.files[p], re.M)[:n]
        return f"{p}  [{', '.join(defs)}]"
