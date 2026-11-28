"""Faisabilité du routeur appris (contribution A).

Question : après le correcteur gratuit (R2 + schéma), peut-on prédire si l'appel au LLM
(stratégie hybride R2 -> R1full) va réussir, et ainsi économiser du temps LLM sur les cas perdus ?

Unité : un pipeline (tâche, modèle, run) qui échoue encore après R2schema.
Politique « porte apprise » : appeler le LLM seulement si P(succès LLM) >= seuil.
Comparaisons : jamais appeler (R2schema seul), toujours appeler (cascade fixe de MLWA),
oracle (appeler seulement quand le LLM réussit).
Validation : leave-one-task-out (apprendre sur 2 tâches, tester sur la 3e).

Approximation héritée de analyze_cascade.py : l'hybride part de la sortie de R2, pas de R2schema.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'repairs')
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'distill')
os.makedirs(OUT, exist_ok=True)
TASKS = ['aq', 'fraud', 'traffic']
MODELS = ['qwen3b', 'qwen7b', 'llama8b', 'mistral7b']


def load(t, m, s):
    p = os.path.join(BASE, f'{t}_{m}_{s}', 'results.json')
    if not os.path.exists(p):
        return None
    return {r['run']: r for r in json.load(open(p, encoding='utf-8'))['runs']}


def last_round(r):
    return r['rounds'][-1] if r.get('rounds') else {}


import re
EXC = ['NameError', 'AttributeError', 'KeyError', 'ValueError', 'TypeError', 'ImportError',
       'ModuleNotFoundError', 'AssertionError', 'FileNotFoundError', 'IndexError', 'SyntaxError']


def rich_features(t, m, k, rs, py):
    tail = py.get('tail', '') or ''
    f = {f'exc_{e}': len(re.findall(rf'\b{e}\b', tail)) for e in EXC}
    f['err_in_test'] = int('test_pipeline.py:' in tail)
    f['err_in_pipeline'] = int('pipeline.py:' in tail.replace('test_pipeline.py:', ''))
    f['n_distinct_exc'] = sum(1 for e in EXC if f[f'exc_{e}'] > 0)
    rd = os.path.join(BASE, f'{t}_{m}_R2schema', f'run_{k:02d}')
    rounds = sorted([d for d in os.listdir(rd) if d.startswith('round_')]) if os.path.isdir(rd) else []
    loc_p = loc_t = 0
    if rounds:
        last = os.path.join(rd, rounds[-1])
        for fn, key in [('pipeline.py', 'p'), ('test_pipeline.py', 't')]:
            fp = os.path.join(last, fn)
            if os.path.exists(fp):
                n = sum(1 for _ in open(fp, encoding='utf-8', errors='ignore'))
                if key == 'p': loc_p = n
                else: loc_t = n
    f['loc_pipeline'] = loc_p
    f['loc_test'] = loc_t
    return f


rows = []
for t in TASKS:
    for m in MODELS:
        sc, hy = load(t, m, 'R2schema'), load(t, m, 'R2R1full')
        if sc is None or hy is None:
            print('manquant', t, m); continue
        for k, rs in sc.items():
            h = hy[k]
            lr = last_round(rs)
            py = lr.get('pytest', {}) or {}
            st = lr.get('strength', {}) or {}
            stat = lr.get('static', {}) or {}
            func = rs.get('functional_final') or {}
            rows.append(dict(
                task=t, model=m, run=k,
                pass0=rs['pass_round0'],
                schema_pass=rs['pass_final'],
                schema_func=bool(func.get('functional_ok')),
                hyb_pass=h['pass_final'],
                hyb_func=bool((h.get('functional_final') or {}).get('functional_ok')),
                hyb_sec=h['repair_seconds'],
                # indices observables après l'étape gratuite
                hint0=rs.get('hint_round0') or 'none',
                hint=rs.get('hint_final') or 'none',
                n_pass=py.get('passed', 0) or 0, n_fail=py.get('failed', 0) or 0,
                n_err=py.get('errors', 0) or 0,
                tests=st.get('tests', 0) or 0, asserts=st.get('asserts', 0) or 0,
                rvh=bool(rs.get('rvh_final')),
                parses=bool((stat.get('pipeline') or {}).get('parses', True)) and bool((stat.get('test') or {}).get('parses', True)),
                func_now=bool(func.get('functional_ok')),
                **rich_features(t, m, k, rs, py),
            ))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'router_dataset.csv'), index=False)
print(f'{len(df)} pipelines ; passent d\'emblée {df.pass0.sum()} ; passent après R2schema {df.schema_pass.sum()}')
fail = df[~df.schema_pass].copy()
print(f'Encore en échec après R2schema : {len(fail)} ; le LLM (hybride) en répare {fail.hyb_pass.sum()} '
      f'({fail.hyb_pass.mean():.0%}) pour {fail.hyb_sec.sum()/3600:.1f} h')
print(fail.groupby('task')[['hyb_pass']].agg(['count', 'sum']))
print('\nTaux de succès du LLM par indice (après R2schema) :')
print(fail.groupby('hint').hyb_pass.agg(['count', 'mean']).sort_values('count', ascending=False))

# ---- features ----
CAT = ['model', 'hint', 'hint0']
NUM = ['n_pass', 'n_fail', 'n_err', 'tests', 'asserts', 'rvh', 'parses', 'func_now',
       'err_in_test', 'err_in_pipeline', 'n_distinct_exc', 'loc_pipeline', 'loc_test'] + [f'exc_{e}' for e in EXC]
if os.environ.get('SIMPLE'):
    NUM = NUM[:8]


def featurize(d, cols=None):
    X = pd.get_dummies(d[CAT].astype(str)).join(d[NUM].astype(float))
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0)
    return X


def evaluate(d, call):
    """d = toutes les lignes d'une tâche ; call = masque booléen sur les lignes en échec."""
    base_pass = d.schema_pass.sum()
    f = d[~d.schema_pass]
    p = base_pass + f.hyb_pass[call].sum()
    fu = d.schema_func[d.schema_pass].sum() + f.hyb_func[call].sum()
    h = f.hyb_sec[call].sum() / 3600
    return p, fu, h, int(call.sum())


res = []
trees = {}
for test_t in TASKS:
    tr = fail[fail.task != test_t]; te_all = df[df.task == test_t]; te = fail[fail.task == test_t]
    Xtr = featurize(tr); Xte = featurize(te, Xtr.columns)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, class_weight='balanced', random_state=42)
    clf.fit(Xtr, tr.hyb_pass)
    trees[test_t] = export_text(clf, feature_names=list(Xtr.columns))
    prob = clf.predict_proba(Xte)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(te))
    for name, call in [('jamais (R2schema seul)', np.zeros(len(te), bool)),
                       ('toujours (cascade MLWA)', np.ones(len(te), bool)),
                       ('oracle', te.hyb_pass.values)]:
        res.append((test_t, name, *evaluate(te_all, pd.Series(call, index=te.index))))
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        call = pd.Series(prob >= thr, index=te.index)
        res.append((test_t, f'arbre seuil {thr}', *evaluate(te_all, call)))

R = pd.DataFrame(res, columns=['tâche test', 'politique', 'passent', 'fonctionnent', 'heures LLM', 'appels LLM'])
R.to_csv(os.path.join(OUT, 'router_loto.csv'), index=False)
tot = R.groupby('politique', sort=False)[['passent', 'fonctionnent', 'heures LLM', 'appels LLM']].sum()
print('\n=== Leave-one-task-out, total des 3 tâches ===')
print(tot.round(1).to_string())
print('\n=== Par tâche ===')
print(R.round(1).to_string(index=False))
with open(os.path.join(OUT, 'trees.txt'), 'w', encoding='utf-8') as fh:
    for t, s in trees.items():
        fh.write(f'### test = {t}\n{s}\n')
print('\nArbre appris (test = traffic) :\n' + trees['traffic'])
