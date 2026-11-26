"""Final analyses of the MLWA study (26 Nov. 2026), from repairs/*/results.json and runs/*/results.json.

  1. RQ1, air quality: functional check and strength at generation (round 0 of the R2 repair, which re-runs the
     oracles on the original generation).
  2. RQ1: Fisher tests between models, per task (each model against Llama 3.1 8B), Holm-corrected.
  3. RQ2: all pairwise exact McNemar tests between the three strategies run on every cell (R1full, R2, R2+R1full)
     and R2schema, per task, Holm-corrected.
  4. RQ2: logistic GEE (exchangeable working correlation, cluster = generation) of the final pass on strategy,
     model and task, for the strategies run on every cell.
  5. RQ2: residual errors after R2 that R1full repairs (column for tab:residual).
  6. RQ4: rounds used by successful model repairs, model time per repaired pipeline.
Run from results/:  python ../scripts/analyze_final.py  (writes analysis_final.json)
"""
import json, os, re, collections, itertools
import numpy as np
from scipy.stats import fisher_exact, binomtest

HERE = os.getcwd()                              # run from the results/ folder of the repository
REP = os.path.join(HERE, "repairs")
MODELS = ["qwen3b", "qwen7b", "llama8b", "mistral7b"]
TASKS = ["aq", "fraud", "traffic"]
out = {}


def load(task, model, strat):
    p = os.path.join(REP, f"{task}_{model}_{strat}", "results.json")
    if not os.path.exists(p):
        return None
    return {r["run"]: r for r in json.load(open(p, encoding="utf-8"))["runs"]}


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


# 1. air quality at generation
aq = {}
for m in MODELS:
    r2 = load("aq", m, "R2")
    f = sum(1 for r in r2.values() if (r["rounds"][0].get("functional") or {}).get("functional_ok"))
    pf = sum(1 for r in r2.values() if r["pass_round0"] and (r["rounds"][0].get("functional") or {}).get("functional_ok"))
    noassert = sum(1 for r in r2.values() if r["pass_round0"] and r["rounds"][0]["strength"]["asserts"] == 0)
    au = [r["rounds"][0]["functional"]["auprc"] for r in r2.values()
          if (r["rounds"][0].get("functional") or {}).get("functional_ok")]
    aq[m] = dict(n=len(r2), passed=sum(r["pass_round0"] for r in r2.values()), functional=f, pass_and_funct=pf,
                 pass_no_assert=noassert, median_auprc=float(np.median(au)) if au else None)
out["rq1_aq_generation"] = aq

# 2. Fisher between models (pass at generation), against Llama
gen = {}
for t in TASKS:
    for m in MODELS:
        r = load(t, m, "R2")
        gen[(t, m)] = (sum(x["pass_round0"] for x in r.values()), len(r))
fis = []
for t in TASKS:
    a, n = gen[(t, "llama8b")]
    for m in ["qwen3b", "qwen7b", "mistral7b"]:
        b, k = gen[(t, m)]
        fis.append((t, m, fisher_exact([[a, n - a], [b, k - b]])[1]))
adj = holm([p for *_, p in fis])
out["rq1_fisher_vs_llama"] = [dict(task=t, model=m, p=p, p_holm=q) for (t, m, p), q in zip(fis, adj)]

# 3. pairwise McNemar between strategies, per task, pooled over models
STR = ["R1full", "R2", "R2R1full", "R2schema"]
mc = []
for t in TASKS:
    for a, b in itertools.combinations(STR, 2):
        x = y = 0
        for m in MODELS:
            A, B = load(t, m, a), load(t, m, b)
            for k in A:
                if k in B and not A[k]["pass_round0"]:
                    x += A[k]["pass_final"] and not B[k]["pass_final"]
                    y += B[k]["pass_final"] and not A[k]["pass_final"]
        p = binomtest(x, x + y).pvalue if x + y else 1.0
        mc.append((t, a, b, x, y, p))
adj = holm([r[-1] for r in mc])
out["rq2_mcnemar"] = [dict(task=t, a=a, b=b, a_only=x, b_only=y, p=p, p_holm=q) for (t, a, b, x, y, p), q in zip(mc, adj)]

# 4. GEE logistic model
rows = []
for t in TASKS:
    for m in MODELS:
        for s in ["R1full", "R2", "R2R1full"]:
            for k, r in load(t, m, s).items():
                rows.append(dict(y=int(bool(r["pass_final"])), strategy=s, model=m, task=t, gid=f"{t}_{m}_{k}"))
try:
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = pd.DataFrame(rows)
    md = smf.gee("y ~ C(strategy, Treatment('R2')) + C(model, Treatment('llama8b')) + C(task, Treatment('aq'))",
                 "gid", df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    res = md.fit()
    ci = res.conf_int()
    out["rq2_gee"] = {name: dict(OR=float(np.exp(res.params[name])), lo=float(np.exp(ci.loc[name, 0])),
                                 hi=float(np.exp(ci.loc[name, 1])), p=float(res.pvalues[name]))
                      for name in res.params.index}
    out["rq2_gee_n"] = dict(observations=len(df), generations=int(df.gid.nunique()))
except Exception as e:  # noqa
    out["rq2_gee_error"] = repr(e)


# 5. residual errors after R2 repaired by R1full
def cat(tail):
    L = [l for l in tail.splitlines() if l.startswith("E   ")]
    s = " ".join(L) if L else tail
    if re.search(r"convert string to float|could not convert", s): return "categorical"
    if re.search(r"mlflow", s) and re.search(r"ImportError|AttributeError|has no attribute|cannot import", s): return "mlflow_api"
    if re.search(r"KeyError|not in index|Features", s): return "column"
    if re.search(r"NaN|infinity", s): return "nan"
    if re.search(r"ImportError|ModuleNotFoundError|NameError", s): return "import_name"
    if re.search(r"assert|AssertionError|Failed", s): return "assertion"
    return "other"


resid = {}
for t in TASKS:
    c = collections.defaultdict(lambda: collections.Counter())
    for m in MODELS:
        r2, hy, r1 = load(t, m, "R2"), load(t, m, "R2R1full"), load(t, m, "R1full")
        for k, r in r2.items():
            if r["pass_final"]:
                continue
            g = cat(r["rounds"][-1]["pytest"]["tail"])
            c[g]["fail"] += 1
            c[g]["hybrid"] += bool(hy[k]["pass_final"])
            c[g]["r1full"] += bool(r1[k]["pass_final"])
    resid[t] = {g: dict(v) for g, v in c.items()}
out["rq2_residual"] = resid

# 6. rounds and cost
cost = {}
for t in TASKS:
    for s in ["R1full", "R2R1full"]:
        byround = collections.Counter(); secs = 0.0; rep = 0; per_fail = []
        for m in MODELS:
            for k, r in load(t, m, s).items():
                if r["pass_round0"]:
                    continue
                secs += r["repair_seconds"] or 0
                if r["repair_seconds"]:
                    per_fail.append(r["repair_seconds"])
                if r["pass_final"]:
                    rep += 1
                    if s == "R2R1full" and r.get("pass_after_R2_phase"):
                        byround[0] += 1
                    else:
                        byround[r["rounds_used"]] += 1
        cost[f"{t}_{s}"] = dict(repaired=rep, model_hours=secs / 3600, minutes_per_repaired=secs / 60 / rep if rep else None,
                                median_minutes_per_model_call_pipeline=float(np.median(per_fail)) / 60 if per_fail else None,
                                repaired_by_rounds={str(k): v for k, v in sorted(byround.items())})
out["rq4_cost"] = cost

json.dump(out, open(os.path.join(HERE, "analysis_final.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
