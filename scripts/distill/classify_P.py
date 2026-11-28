"""Familles de défauts des échecs après R2schema sur les générations P (validation), et ce que R3 en répare."""
import os, pandas as pd, importlib.util, sys
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
src = open(os.path.join(HERE, 'distill_classify.py'), encoding='utf-8').read().split("\nd = pd.read_csv")[0]
ns = {'__file__': os.path.join(HERE, 'distill_classify.py')}; exec(src, ns)
D = pd.read_csv(os.path.join(OUT, 'validation_P_per_pipeline.csv'))
F = D[~D.R2s_p].copy()
F['error'] = [ns['tail_of'](r.task, r.model, int(r.run), 'P_R2schema') for r in F.itertuples()]
F['family'] = [ns['family'](e)[0] for e in F.error]
F.to_csv(os.path.join(OUT, 'validation_P_families.csv'), index=False)
g = F.groupby('family').agg(cas=('run', 'size'), R3=('R3_p', 'sum'), R3_fonct=('R3_f', 'sum'))
print(f'{len(F)} échecs après R2schema'); print(g.to_string())
print(pd.crosstab(F.family, F.task).to_string())
d = pd.read_csv(os.path.join(OUT, 'distill_families.csv'))
print('\nRappel conception :'); print(d.family.value_counts().sort_index().to_string())
