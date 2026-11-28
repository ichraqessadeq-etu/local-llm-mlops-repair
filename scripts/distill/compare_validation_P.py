"""Validation de R3 sur données non vues : les 240 générations du prompt préventif (P) de MLWA.
Compare, pipeline par pipeline : génération brute (round 0), R2schema, R3 ; test de McNemar apparié
R3 contre R2schema (passent, et fonctionnent) ; détail par tâche et par règle déclenchée."""
import json, os, re
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', '..', 'results', 'repairs')
OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
TASKS, MODELS = ['aq', 'fraud', 'traffic'], ['qwen3b', 'qwen7b', 'llama8b', 'mistral7b']


def load(name):
    p = os.path.join(BASE, name, 'results.json')
    if not os.path.exists(p):
        return None
    return {r['run']: r for r in json.load(open(p, encoding='utf-8'))['runs']}


rows, absent = [], []
for t in TASKS:
    for m in MODELS:
        a, b = load(f'{t}_{m}_P_R2schema'), load(f'{t}_{m}_P_R3')
        if a is None or b is None:
            absent.append(f'{t}_{m}'); continue
        for k in a:
            if k not in b:
                continue
            ra, rb = a[k], b[k]
            notes = [n for x in rb.get('rounds', []) for n in (x.get('notes') or [])]
            rules = sorted({lab for n in notes for lab, pat in
                            [('F1', r'catalogue|n\'existe pas|mot-clé'), ('F2', r'module .* inexistant'),
                             ('F3', r'features du schéma|depuis le schéma'), ('F6', r'cible imposée')]
                            if re.search(pat, n)})
            rows.append(dict(task=t, model=m, run=k, brut=bool(ra['pass_round0']),
                             R2s_p=bool(ra['pass_final']), R2s_f=bool((ra.get('functional_final') or {}).get('functional_ok')),
                             R3_p=bool(rb['pass_final']), R3_f=bool((rb.get('functional_final') or {}).get('functional_ok')),
                             R2s_w=bool(ra.get('weakened_final')), R3_w=bool(rb.get('weakened_final')),
                             regles=','.join(rules)))
if absent:
    print('Séries manquantes (calcul pas fini ?) :', absent)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, 'validation_P_per_pipeline.csv'), index=False)


def mcnemar(x, y):
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def summary(d, lab):
    b, c, p = mcnemar(d.R3_p, d.R2s_p)
    bf, cf, pf = mcnemar(d.R3_f, d.R2s_f)
    return dict(groupe=lab, n=len(d), brut=int(d.brut.sum()), R2schema_passent=int(d.R2s_p.sum()),
                R3_passent=int(d.R3_p.sum()), R2schema_fonct=int(d.R2s_f.sum()), R3_fonct=int(d.R3_f.sum()),
                R3_gagne=b, R3_perd=c, p_passent=round(p, 4), p_fonct=round(pf, 4),
                affaiblis_R3=int(d.R3_w.sum()))


S = pd.DataFrame([summary(D, 'total')] + [summary(D[D.task == t], t) for t in TASKS if (D.task == t).any()])
print(S.to_string(index=False))
S.to_csv(os.path.join(OUT, 'validation_P_summary.csv'), index=False)
print('\nRègles déclenchées parmi les pipelines que R3 répare et pas R2schema :')
print(D[D.R3_p & ~D.R2s_p].regles.value_counts().to_string())
print('\nPipelines que R2schema répare et que R3 perd :')
print(D[~D.R3_p & D.R2s_p][['task', 'model', 'run', 'regles']].to_string(index=False))
