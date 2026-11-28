"""R3 : correcteur déterministe = R2schema + règles distillées des réparations du LLM (27 nov. 2026).

Origine : analyse des 145 pipelines encore en échec après R2schema (maThese/ROUTEUR/VERDICT_B.md).
Quatre familles de défauts que le LLM répare mal ou cher, et qu'une règle générique peut traiter :

  F1  API inexistante ou mal placée (« cannot import name X from 'mlflow' », « module 'mlflow' has no attribute X »)
      -> si X existe dans un autre module du catalogue (api_catalog.json, construit DANS l'environnement de test),
         on le déplace (mlflow.log_model -> mlflow.sklearn.log_model) ; sinon on retire l'import et on neutralise
         l'appel (instruction remplacée par `pass`, affectation par `None`).
      + mot-clé inconnu (« f() got an unexpected keyword argument 'k' ») -> on retire k des appels à f.
  F2  module inexistant (« No module named 'randomforest' ») -> résolu vers l'estimateur sklearn dont le nom normalisé
      commence par le nom demandé (suffixe préféré : --prefer, « Classifier » par défaut).
  F3  liste de features absente (NameError 'features' / KeyError 'Features') -> définie à partir du schéma :
      toutes les colonnes du fichier de données sauf la cible.
  F6  mauvaise cible (« Unknown label type: 'continuous' ») -> dans preprocess_data, la cible du schéma remplace
      la colonne choisie, et la cible est retirée des features si le DataFrame entier était renvoyé.

Aucune table n'est écrite par tâche : R3 ne connaît que le catalogue de l'environnement et le schéma du CSV.
Les règles ne s'appliquent que si l'erreur pytest correspondante est présente (comme R2schema).
Valeurs attendues des tests (F7) : jamais modifiées.

Usage (mêmes arguments que repair_schema.py) :
  py -3.13 repair_rules.py --src runs\\traffic_qwen7b --data-name traffic.csv --out repairs\\traffic_qwen7b_R3
"""
import argparse, ast, json, re
from pathlib import Path

import repair_loop as rl
import repair_schema as rs

HERE = Path(__file__).resolve().parent
CATALOG = {}
TARGET = {"name": None}
PREFER = {"suffix": "Classifier"}
FEATURE_NAMES = {"features", "feature_cols", "feature_columns", "feature_names", "feature_list", "featurecolumns"}


# ------------------------------------------------------------------------------------------------ outils
def _offsets(code):
    starts, pos = [0], 0
    for line in code.splitlines(True):
        pos += len(line); starts.append(pos)
    return starts


def _abs(starts, lineno, col):
    return starts[lineno - 1] + col


def _parse(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _insert_after_imports(code, block):
    tree = _parse(code)
    last = 0
    if tree:
        for n in tree.body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                last = n.end_lineno
    lines = code.splitlines(True)
    return "".join(lines[:last]) + ("\n" if last and not lines[last - 1].endswith("\n") else "") + block + "".join(lines[last:])


def _ensure_import(code, mod):
    if re.search(r"^\s*import %s\b" % re.escape(mod), code, re.M):
        return code
    return f"import {mod}\n" + code


def _find_in_catalog(name, avoid):
    for mod in ["mlflow", "mlflow.sklearn", "mlflow.tracking", "mlflow.models", "mlflow.pyfunc", "mlflow.artifacts"]:
        if mod != avoid and name in CATALOG["modules"].get(mod, {}) and CATALOG["modules"][mod][name] != "__attr__":
            return mod
    return None


def _neutralize_calls(code, match):
    """Remplace les instructions qui appellent une fonction inexistante : expression -> pass, affectation -> None."""
    tree = _parse(code)
    if not tree:
        return code, 0
    edits = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Expr, ast.Assign)) and isinstance(n.value, ast.Call) and match(n.value.func):
            if isinstance(n, ast.Expr):
                edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, "pass  # R3: API inexistante retirée"))
            else:
                v = n.value
                edits.append((v.lineno, v.col_offset, v.end_lineno, v.end_col_offset, "None  # R3: API inexistante"))
    st = _offsets(code)
    for l1, c1, l2, c2, rep in sorted(edits, reverse=True):
        code = code[:_abs(st, l1, c1)] + rep + code[_abs(st, l2, c2):]
    return code, len(edits)


def _func_matches(func, mod, name):
    chain = ast.unparse(func)
    return chain == f"{mod}.{name}" or chain == name


# ------------------------------------------------------------------------------------------------ F1
def rule_api(code, tail, notes, tag):
    missing = set(re.findall(r"cannot import name '(\w+)' from '(mlflow(?:\.\w+)*)'", tail))
    missing |= {(n, m) for m, n in re.findall(r"module '(mlflow(?:\.\w+)*)' has no attribute '(\w+)'", tail)}
    for name, mod in sorted(missing):
        alt = _find_in_catalog(name, mod)
        # imports « from mod import a, name »
        def sub(mo):
            names = [x.strip() for x in mo.group(1).split(",") if x.strip()]
            keep = [x for x in names if x.split(" as ")[0] != name]
            if len(keep) == len(names):
                return mo.group(0)
            out = [f"from {mod} import " + ", ".join(keep)] if keep else []
            if alt:
                out.append(f"from {alt} import {name}")
            return "\n".join(out) if out else "pass  # R3: import inexistant retiré"
        new = re.sub(r"^from %s import ([^\n(]+)$" % re.escape(mod), sub, code, flags=re.M)
        if alt:
            new2 = re.sub(r"\b%s\.%s\b" % (re.escape(mod), name), f"{alt}.{name}", new)
            if new2 != new:
                new2 = _ensure_import(new2, alt)
            new = new2
            if new != code:
                notes.append(f"{tag}: {mod}.{name} -> {alt}.{name} (catalogue)")
        else:
            new, k = _neutralize_calls(new, lambda f: _func_matches(f, mod, name))
            if new != code:
                notes.append(f"{tag}: {mod}.{name} n'existe pas dans l'environnement : import retiré, {k} appel(s) neutralisé(s)")
        code = new
    for func, kw in sorted(set(re.findall(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", tail))):
        tree = _parse(code)
        if not tree:
            continue
        cuts = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fname = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
                if fname == func:
                    for k in n.keywords:
                        if k.arg == kw:
                            cuts.append((k.lineno, k.col_offset, k.end_lineno, k.end_col_offset))
        st = _offsets(code)
        for l1, c1, l2, c2 in sorted(cuts, reverse=True):
            a, b = _abs(st, l1, c1), _abs(st, l2, c2)
            # retirer aussi la virgule adjacente
            m = re.match(r"\s*,", code[b:])
            if m:
                b += m.end()
            else:
                m2 = re.search(r",\s*$", code[:a])
                if m2:
                    a = m2.start()
            code = code[:a] + code[b:]
        if cuts:
            notes.append(f"{tag}: mot-clé '{kw}' retiré de {len(cuts)} appel(s) à {func}()")
    return code


# ------------------------------------------------------------------------------------------------ F2
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _resolve_estimator(name):
    est = CATALOG.get("sklearn_estimators", {})
    if name in est:
        return name, est[name]
    cands = [n for n in est if _norm(n).startswith(_norm(name)) and _norm(name)]
    if not cands:
        return None, None
    cands.sort(key=lambda n: (not n.endswith(PREFER["suffix"]), len(n)))
    return cands[0], est[cands[0]]


def rule_module(code, tail, notes, tag):
    for mod in sorted(set(re.findall(r"No module named '([\w.]+)'", tail))):
        top = mod.split(".")[0]
        cls, cmod = _resolve_estimator(top)
        if not cls:
            continue

        def sub_from(mo):
            names = [x.strip() for x in mo.group(1).split(",") if x.strip()]
            out = []
            for x in names:
                base, _, alias = x.partition(" as ")
                b = base.strip()
                if b in CATALOG["sklearn_estimators"]:
                    out.append(f"from {CATALOG['sklearn_estimators'][b]} import {x}")
                else:
                    out.append(f"from {cmod} import {cls} as {alias.strip() or b}")
            return "\n".join(out)
        new = re.sub(r"^from %s import ([^\n(]+)$" % re.escape(mod), sub_from, code, flags=re.M)
        new = re.sub(r"^import %s(\s+as\s+(\w+))?\s*$" % re.escape(mod),
                     lambda mo: f"import {cmod} as {mo.group(2) or top}", new, flags=re.M)
        if new != code:
            notes.append(f"{tag}: module '{mod}' inexistant -> {cmod} ({cls}) (catalogue sklearn)")
            code = new
    return code


# ------------------------------------------------------------------------------------------------ F3
def _features_list():
    return [c for c in rs.SCHEMA["columns"] if c != TARGET["name"]]


def rule_features(code, tail, notes, tag):
    # « name 'X' is not defined » : aussi quand le pipeline enveloppe le NameError dans sa propre exception
    undefined = {n for n in re.findall(r"name '(\w+)' is not defined", tail) if n.lower() in FEATURE_NAMES}
    undefined = {n for n in undefined if re.search(r"\b%s\b" % n, code)
                 and not re.search(r"^\s*%s\s*=" % n, code, re.M)}
    if undefined:
        block = "".join(f"{n} = {_features_list()!r}  # R3: features du schéma (toutes les colonnes sauf la cible)\n"
                        for n in sorted(undefined))
        code = _insert_after_imports(code, block)
        notes.append(f"{tag}: défini {', '.join(sorted(undefined))} depuis le schéma")
    keys = {k for k in re.findall(r"KeyError: [\"'](\w+)[\"']", tail) if k.lower() in FEATURE_NAMES}
    for k in sorted(keys):
        pat = r"\[(['\"])%s\1\]" % re.escape(k)
        if re.search(pat, code):
            code = re.sub(pat, "[_R3_FEATURES]", code)
            if "_R3_FEATURES =" not in code:
                code = _insert_after_imports(code, f"_R3_FEATURES = {_features_list()!r}  # R3: features du schéma\n")
            notes.append(f"{tag}: colonne fictive '{k}' remplacée par la liste des features du schéma")
    return code


# ------------------------------------------------------------------------------------------------ F6
def rule_target(code, tail, notes, tag):
    if not re.search(r"Unknown label type", tail):
        return code
    tree = _parse(code)
    if not tree:
        return code
    tgt = TARGET["name"]
    edits = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess_data"]:
        df = fn.args.args[0].arg if fn.args.args else "data"
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple) and len(n.value.elts) == 2:
                X, y = n.value.elts
                if (isinstance(y, ast.Subscript) and isinstance(y.slice, ast.Constant)
                        and isinstance(y.slice.value, str) and y.slice.value != tgt):
                    edits.append((y.slice.lineno, y.slice.col_offset, y.slice.end_lineno, y.slice.end_col_offset, repr(tgt)))
                    if isinstance(X, ast.Name) and X.id == df:
                        edits.append((X.lineno, X.col_offset, X.end_lineno, X.end_col_offset,
                                      f"{df}.drop(columns=[{tgt!r}])"))
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("y", "target", "labels") and isinstance(n.value, ast.Subscript)
                    and isinstance(n.value.slice, ast.Constant) and isinstance(n.value.slice.value, str)
                    and n.value.slice.value != tgt):
                s = n.value.slice
                edits.append((s.lineno, s.col_offset, s.end_lineno, s.end_col_offset, repr(tgt)))
    st = _offsets(code)
    for l1, c1, l2, c2, rep in sorted(set(edits), reverse=True):
        code = code[:_abs(st, l1, c1)] + rep + code[_abs(st, l2, c2):]
    if edits:
        notes.append(f"{tag}: cible imposée par le schéma ({tgt!r}), {len(edits)} modification(s) dans preprocess_data")
    return code


# ------------------------------------------------------------------------------------------------ R3
def r3_repair(pipe, test, rec):
    pipe, test, notes = rs.r2schema_repair(pipe, test, rec)
    tail = (rec["pytest"] or {}).get("tail", "") or ""
    files = {"pipeline": pipe or "", "test": test or ""}
    for k in files:
        code = files[k]
        code = rule_api(code, tail, notes, k)
        code = rule_module(code, tail, notes, k)
        code = rule_features(code, tail, notes, k)
        if k == "pipeline":
            code = rule_target(code, tail, notes, k)
        files[k] = code
    return files["pipeline"], files["test"], notes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-name", required=True)
    ap.add_argument("--target", help="colonne cible (par défaut : la dernière colonne du CSV)")
    ap.add_argument("--prefer", default="Classifier")
    ap.add_argument("--catalog", default=str(HERE / "api_catalog.json"))
    ap.add_argument("--runs")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--pytest-python", default=rl.m2.DEFAULT_PYTEST_PY)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    CATALOG.update(json.loads(Path(a.catalog).read_text()))
    rs.SCHEMA.update(rs.schema_from_csv(Path(a.src) / a.data_name))
    TARGET["name"] = a.target or rs.SCHEMA["columns"][-1]
    PREFER["suffix"] = a.prefer
    rl.r2_repair = r3_repair
    a.strategy, a.backend, a.model = "R2", "none", None
    rl.main(a)
    res = Path(a.out) / "results.json"
    d = json.loads(res.read_text(encoding="utf-8"))
    d["meta"].update(strategy="R3", schema=rs.SCHEMA, target=TARGET["name"], prefer=a.prefer,
                     catalog_versions=CATALOG.get("versions"), rules_script=rl.sha256_file(__file__),
                     catalog_sha256=rl.sha256_file(a.catalog))
    res.write_text(json.dumps(d, indent=2), encoding="utf-8")
