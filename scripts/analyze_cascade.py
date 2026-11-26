import json, os
M=['qwen3b','qwen7b','llama8b','mistral7b']
def load(t,m,s):
    p=f'repairs/{t}_{m}_{s}/results.json'
    if not os.path.exists(p): return None
    return {r['run']:(r['pass_final'], bool((r.get('functional_final') or {}).get('functional_ok')), r['repair_seconds']) for r in json.load(open(p))['runs']}
for t in ['aq','fraud','traffic']:
    res={}
    n=0; ok=True
    agg={}
    for m in M:
        S={s:load(t,m,s) for s in ['R2','R2schema','R1full','R2R1full']}
        if any(v is None for k,v in S.items() if k!='R2schema'): ok=False; continue
        has_schema=S['R2schema'] is not None
        for k in S['R2'].keys():
            n+=1
            def add(name,p,f,c):
                a=agg.setdefault(name,[0,0,0.0]); a[0]+=p; a[1]+=f; a[2]+=c
            for s in ['R2','R1full','R2R1full']+(['R2schema'] if has_schema else []):
                p,f,c=S[s][k]; add(s,p,f,c)
            # cascades (offline): stop at first stage that passes
            r2=S['R2'][k]; hy=S['R2R1full'][k]
            p,f,c=(r2 if r2[0] else hy); add('casc R2>LLM(hyb)',p,f,c)
            if has_schema:
                sc=S['R2schema'][k]
                p,f,c=(sc if sc[0] else hy); add('casc R2schema>LLM(hyb)',p,f,c)
            # oracle any
            anyp=any(S[s][k][0] for s in S if S[s] is not None); add('any strategy',anyp,0,0)
    print(f'== {t} n={n}', '' if ok else '(incomplete models skipped)')
    for name,(p,f,c) in agg.items(): print(f'  {name:26s} pass {p:4d}  funct {f:4d}  LLM {c/3600:6.1f} h')
