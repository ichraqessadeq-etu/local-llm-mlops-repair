# Execution-guided repair of LLM-generated MLOps pipelines with local open-weight models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24551862.svg)](https://doi.org/10.5281/zenodo.24551862)

Code, generations, repairs and analysis scripts of the study
*From Model Repairs to Rules: Execution-Guided Self-Healing of LLM-Generated MLOps Pipelines with Locally Run
Open-Weight Models* (I. Essadeq, S. Nouh, K. Kandali; manuscript submitted to *Array*). Version 2.0.0 adds the
rules distilled from the model's repairs (R3), their validation on unseen generations and the analyses of the
distillation; version 1.0.0 matches an earlier version of the manuscript without R3.

Four open-weight models, quantised to 4 bits and served by [Ollama](https://ollama.com) on a laptop without a GPU
(Intel Core i7-8665U, 16 GB RAM), generate an MLOps pipeline (`pipeline.py`, `test_pipeline.py`, a GitHub Actions
workflow) for three smart-city tasks. Every pipeline is executed in a pinned environment, and the failing ones are
repaired with seven strategies, judged by four oracles (execution, static detectors, test strength, independent
functional harness).

| Task | Data | Generations |
|---|---|---|
| Air quality | UCI Air Quality (De Vito et al., 2008) | 20 per model + 20 with the preventive prompt |
| Fraud | ULB credit-card fraud, OpenML 1597 (Dal Pozzolo et al., 2015), 10,000-row subsample | 30 per model + 20 preventive |
| Traffic | UCI Metro Interstate Traffic Volume (Hogue, 2019), hourly congestion label | 30 per model + 20 preventive |

Models (Ollama tags, Q4_K_M): `qwen2.5-coder:3b`, `qwen2.5-coder:7b`, `llama3.1:8b`, `mistral:7b`.

| Strategy | Feedback to the model |
|---|---|
| R0 | "does not work, fix it" (no diagnostic) |
| R1static | RVH/IIH violations found by the static detectors |
| R1exec | last 40 lines of the pytest output + pinned versions |
| R1full | both |
| R2 | none: deterministic corrector, no model |
| R2+R1full | R2 first, then R1full |
| R2schema | R2 + fixes that read the data schema (encoding of text columns, imputation, column names), no model |
| R3 | none: R2schema + four rules distilled from the model's successful repairs (MLflow API, module, features, label) |
| P | no repair: two preventive sentences added to the generation prompt |

## Main results

| | Air quality (80) | Fraud (120) | Traffic (120) |
|---|---|---|---|
| Pass at generation | 16 | 18 | 17 |
| R2 (no model) | 46 | 68 | 29 |
| R1full (model alone) | 40 | 51 | 43 |
| R2+R1full (hybrid) | **61** | **87** | 50 |
| R2schema (no model) | 46 | 68 | 61 |
| Cascade R2schema → model (simulated) | 61 | 87 | 72 |
| R3 (no model) | 54 | 86 | 83 |
| Cascade R3 → model (simulated) | **62** | **95** | **87** |

R3 makes 223 of the 320 pipelines pass without any model call (cascade R2schema → model: 220, 24.5 h of model
time) and yields 163 working detectors (145). On the 240 preventive generations, not used to design the rules,
R3 makes 106 pass against 95 for R2schema (11 discordant pairs to 0, exact McNemar p = 0.001).
Preventive prompt: 31/240 pass, against 51/320 with the base prompt (p = 0.34).
Feedback ablation (air quality, 25 failures): R0 0, R1static 0, R1exec 17, R1full 15.

## Layout

```
scripts/generate.py               generation for the three tasks (--task aq|fraud|traffic, --variant base|preventive)
scripts/icista_m2_benchmark.py    extraction, pytest execution, RVH/IIH detectors (air-quality series, unchanged)
scripts/detectors.py              RVH / IIH static detectors
scripts/repair_loop.py            repair loop: R0, R1static, R1exec, R1full, R2, R2+R1full, oracles, functional harness
scripts/icais_r2_corrector.py     deterministic corrector R2
scripts/repair_schema.py          R2schema (R2 + data schema), no model
scripts/repair_rules.py           R3 = R2schema + rules distilled from the model's repairs, no model
scripts/build_api_catalog.py      catalogue of MLflow / scikit-learn names and parameters, to run with the pinned
                                  Python 3.7 environment; output: scripts/api_catalog.json (read by repair_rules.py)
scripts/export_repairs.py         copies a repair series into results/repairs/ without MLflow runs or data copies
scripts/distill/                  analyses of the distillation (outputs in results/distill/, labelled in French):
                                    distill_extract.py      diffs between R2 and the model's successful repairs
                                    distill_classify.py     families F1-F10 of the 145 failures left by R2schema
                                    compare_r3.py           R3 against the other strategies (Table "R3")
                                    compare_validation_P.py R3 against R2schema on the 240 preventive generations
                                    classify_P.py, why_f1_P.py  families and stacked defects of those generations
                                    stats_r3_article.py     McNemar tests and pass@k
                                    router_feasibility.py   decision tree deciding when to call the model
scripts/prepare_fraud.py          fraud subsample from OpenML 1597 (MD5 checked, seed 42)
scripts/prepare_traffic.py        traffic data and congestion label
scripts/analyze_final.py          statistics of the paper (Fisher, Holm, McNemar, GEE, residual errors, rounds, cost)
scripts/analyze_cascade.py        yield and cost of the strategies and of the cascade (Table "cascade")
scripts/analyze_prevention.py     base vs preventive prompt (Table "prevention")
results/runs/<task>_<model>[_P]/  generations: results.json (settings, hashes, Ollama model details, summary) and
                                  run_XX/ with raw_llm_output.txt, pipeline.py, test_pipeline.py, analysis.json
results/repairs/<task>_<model>_<strategy>/
                                  repairs: results.json (per-run outcome, rounds, oracles, model time) and
                                  run_XX/round_K/ with the files of each round, the feedback and the model's answer
results/analysis_final.json       output of analyze_final.py
data/air_quality.csv              cleaned and labelled UCI Air Quality data (7,258 hours)
data/traffic.csv                  prepared traffic data (40,564 hours), traffic_info.json
data/fraud_info.json              size, prevalence and SHA-256 of the fraud subsample (see below)
```

`aq` is the air-quality task. The air-quality base generations (`results/runs/aq_<model>`) are the series first
released in [local-llm-mlops-airquality](https://github.com/ichraqessadeq-etu/local-llm-mlops-airquality)
(doi:10.5281/zenodo.24131237); several scripts still call it "ICISTA", the name of a conference submission that was
abandoned. The scripts are kept byte-identical to the versions whose SHA-256 is recorded in every `results.json`, so
their default paths are those of the original machine.

## Reproducing the paper's numbers (no model call)

```bash
pip install -r requirements.txt
cd results
python ../scripts/analyze_final.py        # -> analysis_final.json
python ../scripts/analyze_cascade.py
python ../scripts/analyze_prevention.py
cd ../scripts/distill                     # distillation (R3), outputs in results/distill/
python distill_classify.py && python compare_r3.py && python compare_validation_P.py
python classify_P.py && python stats_r3_article.py && python router_feasibility.py
```

R3 was run on the 145 base generations that fail after R2schema (`<task>_<model>_R3`), on the 175 that pass
after R2schema (`<task>_<model>_R3pass`, to check that the rules do not break them), and, with R2schema, on the
preventive generations (`<task>_<model>_P_R2schema`, `<task>_<model>_P_R3`):

```bash
python repair_rules.py --src runs/traffic_qwen7b --data-name traffic.csv --out repairs/traffic_qwen7b_R3 \
       --pytest-python <path/to/python3.7>
```

## Running new generations and repairs (local, free)

1. Install Ollama and pull the models, e.g. `ollama pull llama3.1:8b`.
2. Prepare the pinned execution environment: Python 3.7.7, pytest 7.4.4, MLflow 1.30.1, scikit-learn 1.0.2,
   pandas 1.3.5, numpy 1.21.6.
3. Prepare the data: `data/traffic.csv` and `data/air_quality.csv` are included. For fraud, download OpenML dataset
   1597 (`creditcard.arff`, MD5 178bcf9bb1f31a3dfe12d0e577884add) and run
   `python scripts/prepare_fraud.py creditcard.arff data/fraud.csv`; the SHA-256 of the result must match
   `data/fraud_info.json`.
4. From `scripts/`, for example:

```bash
python generate.py --backend ollama --model llama3.1:8b --task traffic --n 30 --out runs/traffic_llama8b \
       --pytest-python <path/to/python3.7>
python repair_loop.py --backend ollama --model llama3.1:8b --strategy R2+R1full --src runs/traffic_llama8b \
       --data-name traffic.csv --out repairs/traffic_llama8b_R2R1full --pytest-python <path/to/python3.7>
python repair_schema.py --src runs/traffic_llama8b --data-name traffic.csv --out repairs/traffic_llama8b_R2schema
```

For the air-quality task, pass `--data ../data/air_quality.csv` to `generate.py`. The repair scripts read the data
file from the `--src` folder (`--data-name`): copy `data/<task>.csv` into a generation folder of `results/runs/`
before repairing it. Every script accepts `--resume`: completed runs are kept and an interrupted run is redone. On a CPU-only laptop,
close memory-hungry applications and disable sleep; a full repair series takes from one to eleven hours.

## Not included

- Models' MLflow artefacts (`mlruns/`, pickled models) and the copies of the data in every run folder, which the
  scripts re-create.
- The fraud subsample itself (re-created from OpenML by `prepare_fraud.py`, see above).
- The ten GPT-4o generations used as a reference in the paper, which belong to an earlier study of ours under
  double-blind review; they will be added once that review is complete.

## Data licences

UCI Air Quality and Metro Interstate Traffic Volume: CC BY 4.0 (UCI Machine Learning Repository).
Credit-card fraud: ULB Machine Learning Group, distributed by OpenML (dataset 1597).
Model outputs in `results/` are released for research reproducibility; each model is subject to its own licence.

## Citation and licence

Archived on Zenodo: [10.5281/zenodo.24551862](https://doi.org/10.5281/zenodo.24551862) (this DOI always resolves to the latest version). See `CITATION.cff`. Code: MIT (see `LICENSE`).
