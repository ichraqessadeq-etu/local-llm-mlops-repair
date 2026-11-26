"""Base prompt vs preventive prompt P, per task and model (Table tab:prevention). Run from results/:
    python ../scripts/analyze_prevention.py
"""
import json, os
from scipy.stats import fisher_exact

MODELS = ['qwen3b', 'qwen7b', 'llama8b', 'mistral7b']


def load(p):
    return json.load(open(os.path.join(p, 'results.json'), encoding='utf-8'))


for t in ['aq', 'fraud', 'traffic']:
    tb = tp = nb = np_ = fb = fp = rb = rp = 0
    for m in MODELS:
        sb = load(os.path.join('runs', f'{t}_{m}'))['summary']
        sp = load(os.path.join('runs', f'{t}_{m}_P'))['summary']
        ab, ap = sb.get('all_pass'), sp.get('all_pass')
        pv = fisher_exact([[ab, sb['n'] - ab], [ap, sp['n'] - ap]])[1]
        print(t, m, 'base', ab, '/', sb['n'], 'func', sb.get('functional_ok'), 'rvh', sb.get('rvh'), '| P', ap, '/', sp['n'],
              'func', sp.get('functional_ok'), 'rvh', sp.get('rvh'), 'trunc', sp.get('truncated'), 'p=%.3f' % pv,
              'hints', sp.get('failure_hints'))
        tb += ab; tp += ap; nb += sb['n']; np_ += sp['n']
        fb += sb.get('functional_ok') or 0; fp += sp.get('functional_ok') or 0
        rb += sb.get('rvh') or 0; rp += sp.get('rvh') or 0
    print(t, 'TOTAL base', tb, nb, 'func', fb, 'rvh', rb, '| P', tp, np_, 'func', fp, 'rvh', rp,
          'p=%.4f' % fisher_exact([[tb, nb - tb], [tp, np_ - tp]])[1])
