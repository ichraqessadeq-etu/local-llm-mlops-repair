"""Chiffres de l'article (version Array) pour R3 : McNemar sur les 320, pass@k hors ligne, par tâche.
À relancer après compare_r3.py (qui lit R3pass quand il existe)."""
import json, os
from math import comb
import pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
BASE = os.path.join(HERE, '..', '..', 'results', 'repairs')
D = pd.read_csv(os.path.join(OUT, 'compare_r3_per_pipeline.csv'))

def mc(x, y):
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    return b, c, (binomtest(b, b + c, .5).pvalue if b + c else 1.0)

rows = []
for lab, d in [('all', D)] + [(t, D[D.task == t]) for t in ['aq', 'fraud', 'traffic']]:
    for other in ['R2schema', 'cascMLWA', 'hyb', 'R1full']:
        for k in ['p', 'f']:
            b, c, p = mc(d[f'R3_{k}'], d[f'{other}_{k}'])
            rows.append(dict(groupe=lab, contre=other, mesure={'p': 'passent', 'f': 'fonctionnent'}[k],
                             R3=int(d[f'R3_{k}'].sum()), autre=int(d[f'{other}_{k}'].sum()), R3_gagne=b, R3_perd=c, p=p))
M = pd.DataFrame(rows); M['p'] = M.p.map(lambda v: float(f'{v:.2g}'))
print(M.to_string(index=False)); M.to_csv(os.path.join(OUT, 'stats_r3_mcnemar.csv'), index=False)

def pk(n, c, k): return 1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0
rows = []
for t in ['aq', 'fraud', 'traffic']:
    for m in ['qwen3b', 'qwen7b', 'llama8b', 'mistral7b']:
        R = json.load(open(os.path.join(BASE, f'{t}_{m}_R2schema', 'results.json'), encoding='utf-8'))['runs']
        d = D[(D.task == t) & (D.model == m)]
        n = len(R)
        for lab, c in [('generation', sum(bool(r['pass_round0']) for r in R)), ('R2schema', int(d.R2schema_p.sum())),
                       ('R3', int(d.R3_p.sum()))]:
            for k in [1, 3, 5]:
                rows.append(dict(task=t, model=m, strategie=lab, k=k, passk=pk(n, c, k)))
P = pd.DataFrame(rows)
S = P.groupby(['strategie', 'k']).passk.mean().unstack().mul(100).round(1)
print('\npass@k moyen sur les 12 cellules (%) :'); print(S.to_string())
print(P.groupby(['task', 'strategie', 'k']).passk.mean().unstack().mul(100).round(1).to_string())
P.to_csv(os.path.join(OUT, 'stats_r3_passk.csv'), index=False)
