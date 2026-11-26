# -*- coding: utf-8 -*-
"""Execution-guided self-healing of LLM-generated MLOps pipelines (journal extension of ICISTA 2027).

Starts from saved generations (a runs_* folder of icista_m2_benchmark.py: run_XX/raw_llm_output.txt) and,
for every generation whose tests fail, applies one repair strategy for at most --max-rounds rounds.
Every round is executed with pytest in the pinned environment, checked by the RVH/IIH detectors, checked
for test weakening (fewer tests or assertions than the original), and evaluated functionally by an
independent harness (AUPRC / F1 on a fixed held-out split), so that "the tests pass" is never the only oracle.

Strategies
  R0        naive re-prompt: "the pipeline does not work, fix it" (no diagnostic)            [ICAIS R0]
  R1static  the static detector report (RVH / IIH violations)                                 [ICAIS R1]
  R1exec    the pytest output of the pinned environment
  R1full    pytest output + static detector report
  R2        deterministic corrector, no LLM: seeds (ICAIS R2) + missing imports of known names
            + known non-existent MLflow imports
  R2+R1full hybrid: R2 until it has nothing left to fix, then R1full rounds by the model on what remains
            (the model sees the files as R2 left them; --max-rounds counts the model rounds only)

Examples (from REVUE_SelfHealing_Local/code):
  py -3.13 repair_loop.py --backend fake --strategy R1exec --src ../../ICISTA/code/runs_qwen25coder7b --runs 1,4 --out repairs_dry
  py -3.13 repair_loop.py --backend none --strategy R2     --src ../../ICISTA/code/runs_qwen25coder7b --out repairs/aq_qwen7b_R2
  py -3.13 repair_loop.py --backend ollama --model qwen2.5-coder:7b --strategy R1full \
         --src ../../ICISTA/code/runs_qwen25coder7b --out repairs/aq_qwen7b_R1full
"""
from __future__ import annotations
import argparse, ast, datetime, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import icista_m2_benchmark as m2                     # byte-identical copy of the ICISTA script
from icais_r2_corrector import repair_code as icais_seed_repair   # ICAIS R2 (seeds, removed APIs)

STRATEGIES = ["R0", "R1static", "R1exec", "R1full", "R2", "R2+R1full"]
ENV_DESC = "Python 3.7.7, pytest 7.4.4, MLflow 1.30.1, scikit-learn 1.0.2, pandas 1.3.5, numpy 1.21.6"
ANSWER_FORMAT = ("Return the corrected files in full: first pipeline.py in one ```python block, then "
                 "test_pipeline.py in a second ```python block. Do not omit any function.")
TAIL_LINES = 40                                       # pytest lines given to the model (R1exec, R1full)


def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------
# Feedback messages
# ---------------------------------------------------------------------------------------------------
def static_report(st):
    lines = []
    for name, s in (("pipeline.py", st.get("pipeline")), ("test_pipeline.py", st.get("test"))):
        if not s:
            lines.append(f"- {name}: file missing")
        elif not s.get("parses"):
            lines.append(f"- {name}: syntax error")
        else:
            for v in s.get("rvh_violations", []):
                lines.append(f"- {name}: reproducibility violation: {v}")
            for v in s.get("iih_violations", []):
                lines.append(f"- {name}: infrastructure inconsistency: {v}")
    return "\n".join(lines) if lines else "- no violation found by the static detectors"


def feedback(strategy, rec):
    tail = "\n".join((rec["pytest"].get("tail") or rec["pytest"].get("failure_hint", "")).splitlines()[-TAIL_LINES:])
    if strategy == "R0":
        body = "The pipeline you generated does not work. Fix it."
    elif strategy == "R1static":
        body = ("Static checks of your files against the target environment report:\n" + static_report(rec["static"])
                + "\nFix the files.")
    elif strategy == "R1exec":
        body = (f"Running pytest on your files in the target environment ({ENV_DESC}) gives:\n```\n{tail}\n```\n"
                "Fix the files so that every test passes.")
    elif strategy == "R1full":
        body = (f"Running pytest on your files in the target environment ({ENV_DESC}) gives:\n```\n{tail}\n```\n"
                "Static checks of the same files report:\n" + static_report(rec["static"])
                + "\nFix the files so that every test passes and no violation remains.")
    else:
        raise ValueError(strategy)
    return body + "\n" + ANSWER_FORMAT


# ---------------------------------------------------------------------------------------------------
# Backends (multi-turn)
# ---------------------------------------------------------------------------------------------------
class OllamaChat(m2.OllamaLLM):
    def chat(self, messages):
        t0 = time.time()
        r = self._post("/api/chat", {"model": self.model, "stream": False, "messages": messages,
                                     "options": {"temperature": m2.TEMPERATURE, "num_predict": m2.MAX_TOKENS}},
                       timeout=3600)
        return r["message"]["content"], {"prompt_tokens": r.get("prompt_eval_count", 0),
                                         "completion_tokens": r.get("eval_count", 0),
                                         "truncated": r.get("done_reason") == "length",
                                         "seconds": round(time.time() - t0, 1)}


class FakeChat:
    """Offline dry run: 'repairs' by adding the imports most often missing."""
    model = "fake"

    def check(self):
        return {}

    def chat(self, messages):
        pipe, test, _ = m2.extract(messages[-2]["content"])
        hdr = ("import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n"
               "from sklearn.ensemble import RandomForestClassifier\n")
        text = f"```python\n{pipe or ''}\n```\n```python\n{hdr}{test or ''}\n```\n"
        return text, {"prompt_tokens": 0, "completion_tokens": len(text) // 4, "truncated": False, "seconds": 0.0}


# ---------------------------------------------------------------------------------------------------
# R2: deterministic corrector (no LLM)
# ---------------------------------------------------------------------------------------------------
KNOWN_IMPORTS = {
    "pd": "import pandas as pd", "pandas": "import pandas", "np": "import numpy as np", "numpy": "import numpy",
    "os": "import os", "sys": "import sys", "json": "import json", "logging": "import logging",
    "pytest": "import pytest", "mlflow": "import mlflow", "joblib": "import joblib", "yaml": "import yaml",
    "train_test_split": "from sklearn.model_selection import train_test_split",
    "cross_val_score": "from sklearn.model_selection import cross_val_score",
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "StandardScaler": "from sklearn.preprocessing import StandardScaler",
    "accuracy_score": "from sklearn.metrics import accuracy_score",
    "f1_score": "from sklearn.metrics import f1_score",
    "precision_score": "from sklearn.metrics import precision_score",
    "recall_score": "from sklearn.metrics import recall_score",
    "roc_auc_score": "from sklearn.metrics import roc_auc_score",
    "classification_report": "from sklearn.metrics import classification_report",
    "confusion_matrix": "from sklearn.metrics import confusion_matrix",
    "load_data": "from pipeline import load_data", "preprocess_data": "from pipeline import preprocess_data",
    "train_model": "from pipeline import train_model",
}
# names that do not exist in `mlflow` 1.30 but are commonly imported from it, and their real location
MLFLOW_FIXES = {"log_model": ("mlflow.sklearn", "log_model"), "save_model": ("mlflow.sklearn", "save_model"),
                "load_model": ("mlflow.sklearn", "load_model")}


def _undefined_names(tail):
    return set(re.findall(r"NameError: name '(\w+)' is not defined", tail))


def _statically_undefined(code):
    """Names read but never bound in the file (imports, defs, assignments, arguments, loop/with/except targets)."""
    import builtins
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    bound, used = set(dir(builtins)), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (used if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((al.asname or al.name).split(".")[0] for al in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return used - bound


def _fix_mlflow_import(code):
    """`from mlflow import a, log_model` -> `from mlflow import a` + `from mlflow.sklearn import log_model`."""
    notes = []

    def sub(mo):
        names = [n.strip() for n in mo.group(1).split(",") if n.strip()]
        keep = [n for n in names if n.split(" as ")[0] not in MLFLOW_FIXES]
        moved = [n for n in names if n.split(" as ")[0] in MLFLOW_FIXES]
        if not moved:
            return mo.group(0)
        out = []
        if keep:
            out.append("from mlflow import " + ", ".join(keep))
        for n in moved:
            mod, real = MLFLOW_FIXES[n.split(" as ")[0]]
            out.append(f"from {mod} import {n}")
            notes.append(f"moved {n} to {mod}")
        return "\n".join(out)
    code = re.sub(r"^from mlflow import ([^\n(]+)$", sub, code, flags=re.M)
    return code, notes


def r2_repair(pipe, test, rec):
    """One deterministic round. Returns (pipe, test, notes)."""
    notes = []
    tail = (rec["pytest"] or {}).get("tail", "") or ""
    files = {"pipeline": pipe or "", "test": test or ""}
    for k in files:
        code = files[k]
        code, n = _fix_mlflow_import(code); notes += [f"{k}: {x}" for x in n]
        missing = _undefined_names(tail) | _statically_undefined(code)
        add = [KNOWN_IMPORTS[nm] for nm in sorted(missing) if nm in KNOWN_IMPORTS
               and re.search(r"\b%s\b" % re.escape(nm), code) and KNOWN_IMPORTS[nm] not in code
               and not (k == "pipeline" and KNOWN_IMPORTS[nm].startswith("from pipeline"))]
        if add:
            code = "\n".join(add) + "\n" + code
            notes.append(f"{k}: added {len(add)} import(s): " + "; ".join(add))
        try:
            r = icais_seed_repair(code)
            # ast.unparse reformats everything: keep the result only if the program itself changed
            if ast.dump(ast.parse(r["repaired"])) != ast.dump(ast.parse(code)):
                code = r["repaired"]; notes += [f"{k}: {x}" for x in r["notes"]]
        except Exception as e:                       # the ICAIS corrector needs parsable code
            notes.append(f"{k}: seed repair skipped ({type(e).__name__})")
        files[k] = code
    return files["pipeline"], files["test"], notes


# ---------------------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------------------
def test_strength(test):
    """Number of test functions and of assertions (assert statements + pytest.raises blocks)."""
    try:
        tree = ast.parse(test or "")
    except SyntaxError:
        return {"tests": None, "asserts": None}
    tests = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
                for n in ast.walk(tree))
    asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
    asserts += sum(isinstance(n, ast.With) and "raises" in ast.dump(n) for n in ast.walk(tree))
    return {"tests": tests, "asserts": asserts}


FUNCTIONAL = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
out = {"functional_ok": False}
try:
    from pipeline import load_data, preprocess_data, train_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, f1_score
    X, y = preprocess_data(load_data(DATA))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = train_model(Xtr, ytr)
    if isinstance(m, (tuple, list)):
        m = next(o for o in m if hasattr(o, "predict"))
    s = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.predict(Xte)
    out.update(functional_ok=True, auprc=round(float(average_precision_score(yte, s)), 4),
               f1=round(float(f1_score(yte, m.predict(Xte))), 4), n_test=int(len(yte)))
except BaseException as e:
    out["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
print("FUNCTIONAL_JSON=" + json.dumps(out))
'''


def functional(rd: Path, py: str, data_rel: str):
    code = FUNCTIONAL.replace("DATA", repr(data_rel))
    (rd / "_functional_check.py").write_text(code, encoding="utf-8")
    try:
        p = subprocess.run([py, "_functional_check.py"], cwd=rd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"functional_ok": False, "error": "timeout"}
    m = re.search(r"FUNCTIONAL_JSON=(.*)", p.stdout)
    return json.loads(m.group(1)) if m else {"functional_ok": False, "error": (p.stderr or "")[-300:]}


def run_round(rd: Path, pipe, test, csv: Path, data_name: str, py: str):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "data").mkdir(exist_ok=True)
    shutil.copy(csv, rd / "data" / data_name)
    rec = {"static": {"pipeline": m2.static_checks(pipe), "test": m2.static_checks(test)},
           "strength": test_strength(test)}
    if pipe and test:
        (rd / "pipeline.py").write_text(pipe, encoding="utf-8")
        (rd / "test_pipeline.py").write_text(test, encoding="utf-8")
        rec["pytest"] = m2.run_pytest(rd, py)
        rec["functional"] = functional(rd, py, "data/" + data_name)
    else:
        rec["pytest"] = {"all_pass": False, "failure_hint": "blocks_not_extracted"}
        rec["functional"] = {"functional_ok": False, "error": "blocks_not_extracted"}
    st = [s for s in rec["static"].values() if s and s.get("parses")]
    rec["is_rvh"] = any(s["is_rvh"] for s in st)
    rec["is_iih"] = any(s["is_iih"] for s in st)
    return rec


# ---------------------------------------------------------------------------------------------------
def main(a):
    src, out = Path(a.src), Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        sys.exit(f"'{out}' already contains results; choose another --out (or --resume), nothing was overwritten.")
    if a.resume and (out / "results.json").exists():
        sys.exit(f"'{out}' is already complete (results.json exists).")
    if not Path(a.pytest_python).exists():
        sys.exit(f"pytest interpreter not found: {a.pytest_python} (pass --pytest-python)")
    if a.strategy == "R2" and a.backend != "none":
        sys.exit("R2 is deterministic: use --backend none")
    if a.strategy != "R2" and a.backend == "none":
        sys.exit(f"{a.strategy} needs a model: --backend ollama (or fake for a dry run)")
    csv = src / a.data_name
    if not csv.exists():
        sys.exit(f"data file not found: {csv}")
    prompt = (src / "prompt.txt").read_text(encoding="utf-8")
    runs = sorted(glob.glob(str(src / "run_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    if a.runs:
        keep = {int(x) for x in a.runs.split(",")}
        runs = [r for r in runs if int(r.rsplit("_", 1)[1]) in keep]
    llm = None
    if a.backend == "ollama":
        llm = OllamaChat(a.model); extra = llm.check()
    elif a.backend == "fake":
        llm = FakeChat(); extra = {}
    else:
        extra = {}
    out.mkdir(parents=True, exist_ok=a.resume)
    src_meta = json.loads((src / "results.json").read_text(encoding="utf-8")).get("meta", {})
    meta = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "strategy": a.strategy,
            "backend": a.backend, "repair_model": a.model if a.backend == "ollama" else a.backend,
            "generation_model": src_meta.get("model"), "src": str(src.resolve()), "max_rounds": a.max_rounds,
            "temperature": m2.TEMPERATURE, "max_tokens": m2.MAX_TOKENS, "tail_lines": TAIL_LINES,
            "script_sha256": sha256_file(__file__), "m2_sha256": sha256_file(m2.__file__),
            "data_sha256": sha256_file(csv), "generator_python": platform.python_version(),
            "resumed": bool(a.resume), **extra}
    results = []
    for rpath in runs:
        k = int(rpath.rsplit("_", 1)[1])
        if a.resume and (out / f"run_{k:02d}" / "analysis.json").exists():     # finished before the interruption
            results.append(json.loads((out / f"run_{k:02d}" / "analysis.json").read_text(encoding="utf-8")))
            print(f"run {k:2d}: already done, kept", flush=True)
            continue
        if a.resume and (out / f"run_{k:02d}").exists():                       # interrupted: start it again
            shutil.rmtree(out / f"run_{k:02d}")
        text0 = Path(rpath, "raw_llm_output.txt").read_text(encoding="utf-8")
        pipe, test, _ = m2.extract(text0)
        rd = out / f"run_{k:02d}"
        rounds = [dict(round=0, **run_round(rd / "round_0", pipe, test, csv, a.data_name, a.pytest_python))]
        base = rounds[0]["strength"]
        messages = [{"role": "system", "content": m2.SYSTEM}, {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text0}]
        pass_after_r2_phase = None
        if a.strategy == "R2+R1full":
            # deterministic phase first (free, instantaneous); the LLM only sees what R2 could not fix
            for step in range(1, a.max_rounds + 1):
                if rounds[-1]["pytest"].get("all_pass"):
                    break
                new_pipe, new_test, notes = r2_repair(pipe, test, rounds[-1])
                if new_pipe == (pipe or "") and new_test == (test or ""):
                    break
                pipe, test = new_pipe, new_test
                rr = run_round(rd / f"round_d{step}", pipe, test, csv, a.data_name, a.pytest_python)
                s = rr["strength"]
                rr["weakened"] = (None if base["tests"] is None or s["tests"] is None
                                  else (s["tests"] < base["tests"] or s["asserts"] < base["asserts"]))
                rounds.append({"round": 0, "phase": "R2", "step": step, "notes": notes, "cost": {"seconds": 0.0}, **rr})
            pass_after_r2_phase = bool(rounds[-1]["pytest"].get("all_pass"))
            if len(rounds) > 1:   # the model continues from the files as R2 left them
                messages[2] = {"role": "assistant",
                               "content": f"```python\n{pipe or ''}\n```\n```python\n{test or ''}\n```\n"}
        fb_strategy = "R1full" if a.strategy == "R2+R1full" else a.strategy
        r = 0
        while not rounds[-1]["pytest"].get("all_pass") and r < a.max_rounds:
            r += 1
            prev = rounds[-1]
            if a.strategy == "R2":
                new_pipe, new_test, notes = r2_repair(pipe, test, prev)
                cost, reply = {"seconds": 0.0}, None
                if new_pipe == (pipe or "") and new_test == (test or ""):
                    rounds.append({"round": r, "stopped": "no_applicable_fix", **{k2: prev[k2] for k2 in
                                   ("pytest", "functional", "static", "strength", "is_rvh", "is_iih")}})
                    break
            else:
                msg = feedback(fb_strategy, prev)
                messages.append({"role": "user", "content": msg})
                reply, cost = llm.chat(messages)
                messages.append({"role": "assistant", "content": reply})
                p2, t2, _ = m2.extract(reply)
                notes = [f"returned: pipeline={'yes' if p2 else 'no'}, test={'yes' if t2 else 'no'}"]
                new_pipe, new_test = p2 or pipe, t2 or test        # a file not returned is kept unchanged
            pipe, test = new_pipe, new_test
            rr = run_round(rd / f"round_{r}", pipe, test, csv, a.data_name, a.pytest_python)
            if reply is not None:
                (rd / f"round_{r}" / "llm_output.txt").write_text(reply, encoding="utf-8")
                (rd / f"round_{r}" / "feedback.txt").write_text(messages[-2]["content"], encoding="utf-8")
            s = rr["strength"]
            rr["weakened"] = (None if base["tests"] is None or s["tests"] is None
                              else (s["tests"] < base["tests"] or s["asserts"] < base["asserts"]))
            rounds.append({"round": r, "notes": notes, "cost": cost, **rr})
        final = rounds[-1]
        rec = {"run": k, "pass_round0": bool(rounds[0]["pytest"].get("all_pass")),
               "pass_final": bool(final["pytest"].get("all_pass")),
               "rounds_used": final["round"] - (1 if final.get("stopped") else 0),
               "pass_after_R2_phase": pass_after_r2_phase, "hint_round0": rounds[0]["pytest"].get("failure_hint"),
               "hint_final": final["pytest"].get("failure_hint"), "weakened_final": final.get("weakened"),
               "rvh_round0": rounds[0]["is_rvh"], "rvh_final": final["is_rvh"],
               "functional_final": final["functional"],
               "repair_seconds": round(sum((x.get("cost") or {}).get("seconds", 0) for x in rounds[1:]), 1),
               "rounds": rounds}
        (rd / "analysis.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        f = final["functional"]
        print(f"run {k:2d}: round0 pass={rec['pass_round0']} -> final pass={rec['pass_final']} after "
              f"{rec['rounds_used']} round(s) | hint {rec['hint_round0']} -> {rec['hint_final']} | "
              f"weakened={rec['weakened_final']} | RVH {rec['rvh_round0']}->{rec['rvh_final']} | "
              f"AUPRC={f.get('auprc', '-')} | {rec['repair_seconds']} s", flush=True)
    failing = [x for x in results if not x["pass_round0"]]
    summary = {"n": len(results), "pass_round0": sum(x["pass_round0"] for x in results),
               "failing_round0": len(failing), "repaired": sum(x["pass_final"] for x in failing),
               "repaired_by_round": {r: sum(x["pass_final"] and x["rounds_used"] == r for x in failing)
                                     for r in range(0 if a.strategy == "R2+R1full" else 1, a.max_rounds + 1)},
               "repaired_weakened": sum(bool(x["pass_final"] and x["weakened_final"]) for x in failing),
               "pass_final": sum(x["pass_final"] for x in results),
               "functional_ok_final": sum(bool(x["functional_final"].get("functional_ok")) for x in results),
               "rvh_round0": sum(x["rvh_round0"] for x in results), "rvh_final": sum(x["rvh_final"] for x in results),
               "repair_seconds_total": round(sum(x["repair_seconds"] for x in results), 1)}
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": results}, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"saved -> {out / 'results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, required=True)
    ap.add_argument("--backend", choices=["ollama", "fake", "none"], required=True)
    ap.add_argument("--model", default=None, help="Ollama tag of the repairing model (normally the generating model)")
    ap.add_argument("--src", required=True, help="runs_* folder produced by icista_m2_benchmark.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", help="comma-separated run numbers (default: all)")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--data-name", default="air_quality.csv", help="data file in --src, copied to data/<name>")
    ap.add_argument("--pytest-python", default=m2.DEFAULT_PYTEST_PY)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted --out: keep finished runs, redo the interrupted one")
    main(ap.parse_args())
