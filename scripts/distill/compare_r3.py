"""Comparaison de R3 (règles distillées, sans LLM) aux autres stratégies, sur les 320 pipelines.

Hypothèse (à lever par un run complet) : les 175 pipelines qui passent après R2schema passent aussi avec R3,
puisque R3 = R2schema + des règles qui ne se déclenchent que sur des erreurs précises. R3 n'a été exécuté que sur
les 145 autres.

Stratégies comparées (passent / fonctionnent / heures LLM / tests affaiblis) :
  R2schema ; R1full (LLM seul) ; hybride R2 -> R1full ; cascade R2schema -> hybride (MLWA) ;
  R3 ; cascade R3 -> hybride (le LLM seulement sur ce que R3 ne répare pas).
Plus : sur les 45 succès du LLM, combien R3 récupère gratuitement, et ce que R3 répare que le LLM ne réparait pas.
"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', '..', 'results', 'repairs')
OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
TASKS, MODELS = ['aq', 'fraud', 'traffic'], ['qwen3b', 'qwen7b', 'llama8b', 'mistral7b']


def load(t, m, s):
    p = os.path.join(BASE, f'{t}_{m}_{s}', 'results.json')
    if not os.path.exists(p):
        return {}
    return {r['run']: dict(p=bool(r['pass_final']), f=bool((r.get('functional_final') or {}).get('functional_ok')),
                           s=float(r.get('repair_seconds') or 0), w=bool(r.get('weakened_final')))
            for r in json.load(open(p, encoding='utf-8'))['runs']}


rows, missing = [], []
for t in TASKS:
    for m in MODELS:
        S = {s: load(t, m, s) for s in ['R2schema', 'R1full', 'R2R1full', 'R3', 'R3pass']}
        for k, sc in S['R2schema'].items():
            r3 = S['R3pass'].get(k, sc) if sc['p'] else S['R3'].get(k)   # R3pass : run réel sur les 175 (sinon hypothèse)
            if r3 is None:
                missing.append((t, m, k)); continue
            hy, r1 = S['R2R1full'][k], S['R1full'][k]
            rows.append(dict(task=t, model=m, run=k,
                             R2schema_p=sc['p'], R2schema_f=sc['f'], R2schema_s=0.0, R2schema_w=sc['w'],
                             R1full_p=r1['p'], R1full_f=r1['f'], R1full_s=r1['s'], R1full_w=r1['w'],
                             hyb_p=hy['p'], hyb_f=hy['f'], hyb_s=hy['s'], hyb_w=hy['w'],
                             R3_p=r3['p'], R3_f=r3['f'], R3_s=0.0, R3_w=r3['w'],
                             cascMLWA_p=sc['p'] or hy['p'], cascMLWA_f=sc['f'] if sc['p'] else hy['f'],
                             cascMLWA_s=0.0 if sc['p'] else hy['s'], cascMLWA_w=sc['w'] if sc['p'] else hy['w'],
                             cascR3_p=r3['p'] or hy['p'], cascR3_f=r3['f'] if r3['p'] else hy['f'],
                             cascR3_s=0.0 if r3['p'] else hy['s'], cascR3_w=r3['w'] if r3['p'] else hy['w']))
if missing:
    print(f'ATTENTION : {len(missing)} pipelines sans résultat R3 (run pas fini ?) — exclus :', missing[:8])
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, 'compare_r3_per_pipeline.csv'), index=False)
NAMES = [('R2schema', 'Correcteur + schéma (MLWA)'), ('R1full', 'LLM seul (R1full)'), ('hyb', 'Hybride R2 -> LLM'),
         ('cascMLWA', 'Cascade R2schema -> LLM (MLWA)'), ('R3', 'R3 règles distillées (sans LLM)'),
         ('cascR3', 'Cascade R3 -> LLM')]


def table(d):
    return pd.DataFrame([dict(strategie=lab, passent=int(d[f'{k}_p'].sum()), fonctionnent=int(d[f'{k}_f'].sum()),
                              heures_LLM=round(d[f'{k}_s'].sum() / 3600, 1), tests_affaiblis=int(d[f'{k}_w'].sum()))
                         for k, lab in NAMES])


print(f'=== Total ({len(D)} pipelines) ===')
T = table(D); print(T.to_string(index=False))
T.to_csv(os.path.join(OUT, 'compare_r3_total.csv'), index=False)
for t in TASKS:
    print(f'\n=== {t} ({(D.task == t).sum()}) ===')
    print(table(D[D.task == t]).to_string(index=False))

F = D[~D.R2schema_p]
both = (F.R3_p & F.hyb_p).sum(); only_r3 = (F.R3_p & ~F.hyb_p).sum(); only_llm = (~F.R3_p & F.hyb_p).sum()
print(f'\n=== Les {len(F)} échecs après R2schema ===')
print(f'R3 répare {int(F.R3_p.sum())} (fonctionnels {int(F.R3_f.sum())}) ; le LLM (hybride) {int(F.hyb_p.sum())} '
      f'(fonctionnels {int(F.hyb_f.sum())}, {F.hyb_s.sum()/3600:.1f} h)')
print(f'Les deux : {both} ; R3 seul : {only_r3} ; LLM seul : {only_llm}')
fam = pd.read_csv(os.path.join(OUT, 'distill_families.csv'))[['task', 'model', 'run', 'family']]
F = F.merge(fam, on=['task', 'model', 'run'], how='left')
print('\nPar famille :')
print(F.groupby('family').agg(cas=('run', 'size'), R3=('R3_p', 'sum'), R3_fonct=('R3_f', 'sum'),
                              LLM=('hyb_p', 'sum'), LLM_fonct=('hyb_f', 'sum')).to_string())
