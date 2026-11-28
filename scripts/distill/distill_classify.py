"""Contribution B, étape 2 : classer en familles de défauts les 145 pipelines encore en échec après le
correcteur gratuit (R2schema), à partir de l'erreur pytest du dernier tour de R2schema.
Pour chaque famille : combien de cas, combien le LLM (hybride) en répare, combien deviennent fonctionnels,
et si la famille se prête à une règle déterministe (jugement fondé sur les diffs de distill_cases.md).
"""
import json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', '..', 'results', 'repairs')
OUT = os.path.join(HERE, '..', '..', 'results', 'distill')

# (famille, motif sur l'erreur, règle possible ?, règle envisagée)
FAMILIES = [
    ('F1 API MLflow inexistante', r"mlflow", True,
     "vérifier chaque appel/import mlflow contre le paquet installé (introspection) ; table de remplacement "
     "(log_model -> mlflow.sklearn.log_model, sklearn.log_params -> log_params) ; supprimer les imports inexistants"),
    ('F2 Module inexistant', r"ModuleNotFoundError|No module named", True,
     "résoudre le nom (randomforest, random_forest) vers la classe sklearn la plus proche"),
    ('F3 Variable de features non définie', r"name '[Ff]eatures' is not defined|KeyError: '[Ff]eatures'", True,
     "définir la liste des features à partir du schéma (toutes les colonnes sauf la cible)"),
    ('F4 Colonnes absentes', r"not in index|KeyError: '", True,
     "réconcilier les noms de colonnes avec le schéma (distance d'édition)"),
    ('F5 Variable catégorielle non encodée', r"could not convert string to float", True,
     "encoder les colonnes texte du schéma (get_dummies) avant l'apprentissage"),
    ('F6 Mauvaise cible', r"Unknown label type", True,
     "imposer la colonne cible du schéma"),
    ('F7 Attente du test fausse', r"assert |AssertionError|DID NOT RAISE|has no attribute 'empty'|has no attribute 'is_fitted'", False,
     "modifier la valeur attendue d'un test : décision humaine (sinon on adapte le test au code)"),
    ('F8 Interface entre fonctions', r"takes \d+ positional|unpack|NoneType|missing \d+ required", False,
     "aligner signatures et retours : sémantique, au cas par cas"),
    ('F9 Nom non défini (autre)', r"NameError", False, "dépend du contexte"),
]


def family(err):
    for name, pat, rule, _ in FAMILIES:
        if re.search(pat, err):
            return name, rule
    return 'F10 Autre', False


def tail_of(t, m, run, strat='R2schema'):
    p = os.path.join(BASE, f'{t}_{m}_{strat}', 'results.json')
    r = [x for x in json.load(open(p, encoding='utf-8'))['runs'] if x['run'] == run][0]
    tl = (r['rounds'][-1].get('pytest') or {}).get('tail', '') if r.get('rounds') else ''
    errs = [re.sub(r'\s+', ' ', l.strip()[1:].strip()) for l in tl.splitlines() if l.strip().startswith('E ')]
    if not errs:  # erreurs de collecte : prendre les lignes « ERROR » / exceptions
        errs = [l.strip() for l in tl.splitlines() if re.search(r'Error|Exception', l)][:3]
    return ' || '.join(errs[:4])


d = pd.read_csv(os.path.join(OUT, 'router_dataset.csv'))
fail = d[~d.schema_pass].copy()
fail['error'] = [tail_of(r.task, r.model, int(r.run)) for r in fail.itertuples()]
fam = fail.error.apply(family)
fail['family'] = [f for f, _ in fam]
fail['rule'] = [r for _, r in fam]
fail.to_csv(os.path.join(OUT, 'distill_families.csv'), index=False)

g = fail.groupby('family').agg(cas=('run', 'size'), llm_repare=('hyb_pass', 'sum'),
                               llm_fonctionnel=('hyb_func', 'sum'), heures_llm=('hyb_sec', lambda s: s.sum() / 3600),
                               regle=('rule', 'first')).sort_index()
print(g.round(1).to_string())
R = fail[fail.rule]
print(f"\nFamilles « règle possible » : {len(R)} cas sur {len(fail)} ; le LLM en répare {R.hyb_pass.sum()} "
      f"(sur ses {fail.hyb_pass.sum()} succès), dont {R.hyb_func.sum()} fonctionnels ; "
      f"{R.hyb_sec.sum()/3600:.1f} h de LLM dépensées sur ces familles.")
print('\nPar tâche × famille (cas) :')
print(pd.crosstab(fail.family, fail.task).to_string())
print('\nExemples F10 :')
for e in fail[fail.family == 'F10 Autre'].error.head(12): print('  -', e[:160])
json.dump([dict(famille=n, regle_possible=r, regle=txt) for n, _, r, txt in FAMILIES],
          open(os.path.join(OUT, 'distill_families_def.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
