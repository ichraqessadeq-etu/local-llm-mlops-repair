# -*- coding: utf-8 -*-
# =============================================================================
#  R2 - DETERMINISTIC CORRECTOR  (AST + regex, offline, no LLM, no network)
#  Repairs the RVH / IIH violations flagged by the paper's detectors.
# =============================================================================
from __future__ import annotations
import ast, re
from typing import Optional

# kwarg to inject per estimator/split (mirrors _NEEDS_SEED_CALL in detectors)
_SEED_KW = {
    "train_test_split": "random_state", "RandomForestClassifier": "random_state",
    "RandomForestRegressor": "random_state", "ExtraTreesClassifier": "random_state",
    "GradientBoostingClassifier": "random_state", "DecisionTreeClassifier": "random_state",
    "XGBClassifier": "random_state", "LGBMClassifier": "random_state",
    "KMeans": "random_state", "MLPClassifier": "random_state",
    "KFold": "random_state", "StratifiedKFold": "random_state",
    "ShuffleSplit": "random_state",
}
_SEED_VALUE = 42

# textual API fixes for IIH 'removed_api' / 'incompatible_api'
_TEXT_FIXES = [
    (r"\bnp\.int\b", "int"),
    (r"\bnp\.float\b", "float"),
    (r"\bnp\.bool\b", "bool"),
    (r"\bnp\.object\b", "object"),
    (r"from\s+sklearn\.externals\s+import\s+joblib", "import joblib"),
    (r"\bsklearn\.externals\.joblib\b", "joblib"),
    (r"\.ix\[", ".loc["),
]


def _func_name(node) -> Optional[str]:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class _SeedInjector(ast.NodeTransformer):
    """Add the missing random_state=42 (or .sample(random_state=42))."""
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        name = _func_name(node)
        have = {kw.arg for kw in node.keywords if kw.arg}
        if name in _SEED_KW and _SEED_KW[name] not in have:
            node.keywords.append(ast.keyword(
                arg=_SEED_KW[name], value=ast.Constant(value=_SEED_VALUE)))
        elif name == "sample" and "random_state" not in have:
            node.keywords.append(ast.keyword(
                arg="random_state", value=ast.Constant(value=_SEED_VALUE)))
        return node


def _has_global_seed(code: str) -> bool:
    return bool(re.search(r"(np|numpy)\.random\.seed\s*\(|(^|\W)random\.seed\s*\(",
                          code))


def repair_code(code: str) -> dict:
    """Deterministically repair a code artifact.
    Returns {repaired, changed, notes:[...], residual_unfixable:[...]}."""
    notes, unfixable = [], []
    original = code

    # 1) AST pass: inject seeds into estimators/splits/sample
    try:
        tree = ast.parse(code)
        tree = _SeedInjector().visit(tree)
        ast.fix_missing_locations(tree)
        code = ast.unparse(tree)
        if code != original:
            notes.append("injected random_state into estimators/splits")
    except SyntaxError as e:
        unfixable.append(f"syntax_error: {e}")  # R2 cannot fix; leave for R1

    # 2) text pass: removed/incompatible APIs
    for pat, repl in _TEXT_FIXES:
        if re.search(pat, code):
            code = re.sub(pat, repl, code)
            notes.append(f"replaced /{pat}/ -> {repl}")

    # 3) global seed for stray RNG calls
    if re.search(r"(np|numpy)\.random\.\w+\(", code) and not _has_global_seed(code):
        seed_block = "import numpy as _np_seed\n_np_seed.random.seed(42)\nimport random as _r_seed\n_r_seed.seed(42)\n"
        code = seed_block + code
        notes.append("prepended global seed block")

    return {"repaired": code, "changed": code != original,
            "notes": notes, "residual_unfixable": unfixable}
