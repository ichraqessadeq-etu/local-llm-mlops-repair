"""Contribution B, étape 1 : extraire ce que le LLM a corrigé dans les 45 cas où le correcteur gratuit
(R2schema) échoue et où l'hybride (R2 -> R1full) réussit.

Point de départ du LLM : round_d1 (sortie de R2) ; arrivée : dernier round_N.
Pour chaque cas : l'erreur pytest au départ (lignes « E ») et les changements au niveau AST
(imports, retours de fonctions, assertions, appels de prétraitement, littéraux de colonnes).
Sorties : results/distill_cases.json, results/distill_cases.md (lecture humaine), results/diffs/*.diff
"""
import ast, difflib, json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', '..', 'results', 'repairs')
OUT = os.path.join(HERE, '..', '..', 'results', 'distill')
os.makedirs(os.path.join(OUT, 'diffs'), exist_ok=True)

PREP_CALLS = {'get_dummies', 'LabelEncoder', 'OneHotEncoder', 'fillna', 'dropna', 'astype', 'select_dtypes',
              'to_datetime', 'to_numeric', 'StandardScaler', 'ColumnTransformer', 'replace', 'map', 'drop',
              'SimpleImputer', 'OrdinalEncoder', 'reset_index', 'rename'}


def read(p):
    return open(p, encoding='utf-8', errors='ignore').read() if os.path.exists(p) else ''


def ast_info(src):
    info = dict(imports=set(), funcs={}, asserts=[], calls=set(), strs=set(), parses=True)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        info['parses'] = False
        return info
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: info['imports'].add(a.name)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names: info['imports'].add(f'{n.module}.{a.name}')
        elif isinstance(n, ast.FunctionDef):
            rets = [ast.unparse(r.value) for r in ast.walk(n) if isinstance(r, ast.Return) and r.value is not None]
            info['funcs'][n.name] = dict(args=[a.arg for a in n.args.args], rets=rets)
        elif isinstance(n, ast.Assert):
            info['asserts'].append(ast.unparse(n.test))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name: info['calls'].add(name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 40:
            info['strs'].add(n.value)
    return info


def errors_from_tail(tail):
    lines = [l.strip() for l in tail.splitlines() if l.strip().startswith('E ')]
    out = []
    for l in lines:
        l = re.sub(r'\s+', ' ', l[1:].strip())
        if l and l not in out: out.append(l)
    return out[:6]


d = pd.read_csv(os.path.join(OUT, 'router_dataset.csv'))
cases = d[(~d.schema_pass) & (d.hyb_pass)]
records = []
for _, c in cases.iterrows():
    run_dir = os.path.join(BASE, f'{c.task}_{c.model}_R2R1full', f'run_{int(c.run):02d}')
    res = [r for r in json.load(open(os.path.join(BASE, f'{c.task}_{c.model}_R2R1full', 'results.json'),
                                    encoding='utf-8'))['runs'] if r['run'] == c.run][0]
    det = [r for r in res['rounds'] if r.get('phase') == 'deterministic' or (r.get('cost') or {}).get('seconds') == 0.0 and r.get('notes') and 'returned' not in ' '.join(r.get('notes'))]
    start_round = det[-1] if det else res['rounds'][0]
    start_tail = (start_round.get('pytest') or {}).get('tail', '')
    llm_rounds = sorted([x for x in os.listdir(run_dir) if re.fullmatch(r'round_\d+', x) and x != 'round_0'],
                        key=lambda s: int(s.split('_')[1]))
    start = 'round_d1' if os.path.isdir(os.path.join(run_dir, 'round_d1')) else 'round_0'
    end = llm_rounds[-1]
    rec = dict(task=c.task, model=c.model, run=int(c.run), n_llm_rounds=len(llm_rounds),
               llm_seconds=res['repair_seconds'], weakened=res.get('weakened_final'),
               functional=bool((res.get('functional_final') or {}).get('functional_ok')),
               start_errors=errors_from_tail(start_tail), files={})
    diff_txt = ''
    for fn in ['pipeline.py', 'test_pipeline.py']:
        a, b = read(os.path.join(run_dir, start, fn)), read(os.path.join(run_dir, end, fn))
        A, B = ast_info(a), ast_info(b)
        fd = {}
        fd['loc'] = (a.count('\n'), b.count('\n'))
        fd['similarity'] = round(difflib.SequenceMatcher(None, a, b).ratio(), 2)
        fd['imports_added'] = sorted(B['imports'] - A['imports'])
        fd['imports_removed'] = sorted(A['imports'] - B['imports'])
        fd['calls_added'] = sorted((B['calls'] - A['calls']) & PREP_CALLS)
        fd['calls_removed'] = sorted((A['calls'] - B['calls']) & PREP_CALLS)
        fd['ret_changed'] = {k: (A['funcs'][k]['rets'], B['funcs'][k]['rets']) for k in A['funcs']
                             if k in B['funcs'] and A['funcs'][k]['rets'] != B['funcs'][k]['rets']}
        fd['sig_changed'] = {k: (A['funcs'][k]['args'], B['funcs'][k]['args']) for k in A['funcs']
                             if k in B['funcs'] and A['funcs'][k]['args'] != B['funcs'][k]['args']}
        fd['funcs_added'] = sorted(set(B['funcs']) - set(A['funcs']))
        fd['funcs_removed'] = sorted(set(A['funcs']) - set(B['funcs']))
        fd['asserts_removed'] = [x for x in A['asserts'] if x not in B['asserts']]
        fd['asserts_added'] = [x for x in B['asserts'] if x not in A['asserts']]
        fd['strs_added'] = sorted(B['strs'] - A['strs'])[:12]
        fd['strs_removed'] = sorted(A['strs'] - B['strs'])[:12]
        rec['files'][fn] = fd
        diff_txt += ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                                 f'{start}/{fn}', f'{end}/{fn}', n=1))
    name = f'{c.task}_{c.model}_run{int(c.run):02d}.diff'
    open(os.path.join(OUT, 'diffs', name), 'w', encoding='utf-8').write(diff_txt)
    rec['diff_file'] = name
    rec['diff_lines'] = sum(1 for l in diff_txt.splitlines() if l[:1] in '+-' and not l.startswith(('+++', '---')))
    records.append(rec)

json.dump(records, open(os.path.join(OUT, 'distill_cases.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
with open(os.path.join(OUT, 'distill_cases.md'), 'w', encoding='utf-8') as fh:
    for r in records:
        fh.write(f"## {r['task']} {r['model']} run {r['run']} — {r['n_llm_rounds']} tour(s) LLM, "
                 f"{r['diff_lines']} lignes changées, fonctionnel={r['functional']}, affaibli={r['weakened']}\n")
        for e in r['start_errors']: fh.write(f'- erreur : `{e[:180]}`\n')
        for fn, fd in r['files'].items():
            keep = {k: v for k, v in fd.items() if v not in ([], {}, ()) and k not in ('loc',)}
            fh.write(f'- {fn} {fd["loc"]} : {json.dumps(keep, ensure_ascii=False)[:900]}\n')
        fh.write('\n')
print(len(records), 'cas extraits')
