"""R2schema: the deterministic corrector R2, plus what an operator always knows, the schema of its own data.

Ablation added on 24 Nov. 2026 after the residual-error analysis (paper, Table tab:residual): on traffic, once R2
has fixed the imports, the most frequent remaining error is a text column (weather) passed to the random forest.

R2schema = R2 (unchanged, from repair_loop.py) + three schema-driven fixes, applied only when pytest shows the error:
  (a) text column not encoded ("could not convert string to float"): a shim appended to pipeline.py encodes the
      categorical columns of the data file (categories read from the CSV) at the input of preprocess_data, and every
      remaining text or missing value at the input of train_model and of the fitted model's predict methods;
  (b) missing values reaching the model ("Input contains NaN"): same shim (median imputation, then 0);
  (c) KeyError on a column that is not in the data: the string literal is replaced by the schema column with the
      closest name (difflib ratio >= 0.8, case-insensitive), if there is one.
Nothing is tuned per task: the schema is read from the data file given with --data-name.

Usage (same arguments as repair_loop.py with --strategy R2 --backend none):
  py -3.13 repair_schema.py --src runs\\traffic_qwen7b --data-name traffic.csv --out repairs\\traffic_qwen7b_R2schema
"""
import argparse, difflib, json, re, sys
from pathlib import Path

import pandas as pd

import repair_loop as rl

MARK = "# --- R2schema shim"
SCHEMA = {"columns": [], "categories": {}}

SHIM = r'''

# --- R2schema shim: data schema known to the operator (added by the deterministic corrector, not by the model) ---
import pandas as _r2s_pd
_R2S_CATEGORIES = @@SCHEMA_CATEGORIES@@


def _r2s_encode(X, full=True):
    if not isinstance(X, _r2s_pd.DataFrame):
        return X
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object and c in _R2S_CATEGORIES:
            X[c] = X[c].map({v: i for i, v in enumerate(_R2S_CATEGORIES[c])}).fillna(-1)
        elif X[c].dtype == object and full:
            X[c] = _r2s_pd.to_numeric(X[c], errors="coerce")
    if full:
        num = X.select_dtypes(include="number").columns
        X[num] = X[num].fillna(X[num].median()).fillna(0)
    return X


class _R2SEncoded(object):
    def __init__(self, obj, name):
        self.obj, self.name = obj, name

    def __call__(self, X, *a, **k):
        return getattr(type(self.obj), self.name)(self.obj, _r2s_encode(X), *a, **k)


if "preprocess_data" in globals():
    _r2s_preprocess = preprocess_data

    def preprocess_data(data, *a, **k):
        return _r2s_preprocess(_r2s_encode(data, full=False), *a, **k)

if "train_model" in globals():
    _r2s_train = train_model

    def train_model(X_train, y_train, *a, **k):
        m = _r2s_train(_r2s_encode(X_train), y_train, *a, **k)
        for o in (m if isinstance(m, (tuple, list)) else [m]):
            if hasattr(o, "fit"):
                for n in ("predict", "predict_proba", "decision_function", "score"):
                    if hasattr(type(o), n):
                        try:
                            setattr(o, n, _R2SEncoded(o, n))
                        except Exception:
                            pass
        return m
'''


def schema_from_csv(csv):
    d = pd.read_csv(csv)
    cats = {c: sorted(map(str, d[c].dropna().unique())) for c in d.columns
            if d[c].dtype == object or str(d[c].dtype).startswith("str")}
    return {"columns": list(d.columns), "categories": cats}


def _fix_columns(code, tail):
    notes = []
    for name in sorted(set(re.findall(r"KeyError: [\"'](.+?)[\"']", tail))):
        if name in SCHEMA["columns"]:
            continue
        low = {c.lower(): c for c in SCHEMA["columns"]}
        m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.8)
        if not m:
            continue
        new = low[m[0]]
        pat = r"(['\"])%s\1" % re.escape(name)
        if re.search(pat, code):
            code = re.sub(pat, lambda mo: mo.group(1) + new + mo.group(1), code)
            notes.append(f"column '{name}' -> '{new}' (schema)")
    return code, notes


def r2schema_repair(pipe, test, rec):
    pipe, test, notes = R2_ORIGINAL(pipe, test, rec)
    tail = (rec["pytest"] or {}).get("tail", "") or ""
    for k in ("pipe", "test"):
        code = pipe if k == "pipe" else test
        code, n = _fix_columns(code or "", tail)
        notes += [f"{k}: {x}" for x in n]
        if k == "pipe":
            pipe = code
        else:
            test = code
    if (re.search(r"could not convert string to float|Input contains NaN", tail)
            and pipe and MARK not in pipe and "def train_model" in pipe):
        pipe = pipe.rstrip("\n") + SHIM.replace("@@SCHEMA_CATEGORIES@@", repr(SCHEMA["categories"]))
        notes.append("pipeline: added schema shim (categorical encoding, imputation)")
    return pipe, test, notes


R2_ORIGINAL = rl.r2_repair

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-name", required=True)
    ap.add_argument("--runs")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--pytest-python", default=rl.m2.DEFAULT_PYTEST_PY)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    SCHEMA.update(schema_from_csv(Path(a.src) / a.data_name))
    rl.r2_repair = r2schema_repair                     # main() looks the corrector up at call time
    a.strategy, a.backend, a.model = "R2", "none", None
    rl.main(a)
    res = Path(a.out) / "results.json"
    d = json.loads(res.read_text(encoding="utf-8"))
    d["meta"].update(strategy="R2schema", schema=SCHEMA, schema_script=rl.sha256_file(__file__))
    res.write_text(json.dumps(d, indent=2), encoding="utf-8")
