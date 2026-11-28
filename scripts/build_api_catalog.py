"""Catalogue des API réellement disponibles dans l'environnement figé (pour le correcteur R3).

À lancer avec le Python de l'environnement de test (celui où pytest s'exécute), PAS avec py -3.13 :
  "C:\\Program Files\\Odoo 16.0.20230507\\python\\python.exe" build_api_catalog.py

Pour chaque module inspecté : les noms publics et, pour les appelables, la liste de leurs paramètres
(None si la signature est illisible ou accepte **kwargs). Aussi : les estimateurs sklearn et leur module.
Rien n'est écrit à la main : R3 ne connaît que ce que l'environnement expose réellement.
"""
import importlib, inspect, json, platform, sys

MODULES = ["mlflow", "mlflow.sklearn", "mlflow.tracking", "mlflow.models", "mlflow.pyfunc", "mlflow.artifacts"]


def params_of(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return None                                   # accepte n'importe quel mot-clé
    return [p.name for p in sig.parameters.values() if p.kind != p.VAR_POSITIONAL]


cat = {"python": platform.python_version(), "modules": {}, "versions": {}, "sklearn_estimators": {}}
for name in MODULES:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        cat["modules"][name] = {"_error": repr(e)}
        continue
    entries = {}
    for attr in dir(mod):
        if attr.startswith("_"):
            continue
        try:
            obj = getattr(mod, attr)
        except Exception:
            continue
        entries[attr] = params_of(obj) if callable(obj) and not inspect.isclass(obj) else "__attr__"
    cat["modules"][name] = entries
for lib in ["mlflow", "sklearn", "pandas", "numpy"]:
    try:
        cat["versions"][lib] = importlib.import_module(lib).__version__
    except Exception:
        pass
try:
    from sklearn.utils import all_estimators
    for n, cls in all_estimators():
        cat["sklearn_estimators"][n] = cls.__module__.split("._")[0]
except Exception as e:
    cat["sklearn_estimators"] = {"_error": repr(e)}

out = sys.argv[1] if len(sys.argv) > 1 else "api_catalog.json"
json.dump(cat, open(out, "w"), indent=1, sort_keys=True)
print("catalogue écrit :", out, {k: len(v) for k, v in cat["modules"].items()}, "estimateurs :",
      len(cat["sklearn_estimators"]))
