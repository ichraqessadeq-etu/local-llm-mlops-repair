import os, re, pandas as pd, collections
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
src = open(os.path.join(HERE, 'distill_classify.py'), encoding='utf-8').read().split("\nd = pd.read_csv")[0]
ns = {'__file__': os.path.join(HERE, 'distill_classify.py')}; exec(src, ns)
F = pd.read_csv(os.path.join(OUT, 'validation_P_families.csv'))
X = F[(F.family.str.startswith('F1')) & ~F.R3_p]
c = collections.Counter()
for r in X.itertuples():
    e0 = r.error; e3 = ns['tail_of'](r.task, r.model, int(r.run), 'P_R3')
    c[(ns['family'](e3)[0], re.sub(r"'[^']*'", "'…'", e3)[:110])] += 1
for k, v in c.most_common(15): print(v, k)
print('\nErreurs initiales F1 (R2schema), les plus fréquentes :')
for k, v in collections.Counter(re.sub(r"\s+", " ", e)[:120] for e in X.error.fillna("")).most_common(10): print(v, k)
