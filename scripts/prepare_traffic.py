# -*- coding: utf-8 -*-
"""Traffic task data: UCI Metro Interstate Traffic Volume (Hogue, 2019, doi:10.24432/C5X60B, CC BY 4.0).

Hourly westbound I-94 traffic (Minneapolis-St Paul, 2012-2018) with weather and holidays. Cleaning, documented
in the paper:
  * one row per hour (the raw file repeats an hour when several weather conditions are reported: keep the first);
  * drop physically impossible readings (temperature of 0 K, hourly rain above 100 mm);
  * IsHoliday = 1 for the hours whose `holiday` field is filled;
  * label Congestion = 1 when traffic_volume is at or above its 90th percentile (about 10% of hours, the same kind
    of "anomalous episode" label as the air-quality task); traffic_volume itself is removed (it defines the label).
The feature `weather_main` is kept as text, so that the generated pipelines must encode a categorical variable.

    py -3.13 -I prepare_traffic.py ../data_raw/traffic/Metro_Interstate_Traffic_Volume.csv.gz ../data/traffic.csv
"""
import hashlib, json, sys
from pathlib import Path
import pandas as pd

FEAT = ["temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "IsHoliday", "Hour", "DayOfWeek", "Month"]
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = pd.read_csv(src)
d = raw.drop_duplicates(subset="date_time", keep="first").copy()
n_dedup = len(d)
d = d[(d["temp"] > 0) & (d["rain_1h"] <= 100)].copy()
t = pd.to_datetime(d["date_time"])
d["Hour"], d["DayOfWeek"], d["Month"] = t.dt.hour, t.dt.dayofweek, t.dt.month
d["IsHoliday"] = d["holiday"].notna().astype(int)
thr = float(d["traffic_volume"].quantile(0.90))
d["Congestion"] = (d["traffic_volume"] >= thr).astype(int)
out = d[FEAT + ["Congestion"]].reset_index(drop=True)
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
info = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "source_rows": len(raw),
        "rows_after_dedup": n_dedup, "rows": len(out), "threshold_vehicles_per_hour": thr,
        "prevalence": round(float(out["Congestion"].mean()), 4), "features": FEAT,
        "csv_sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
(dst.parent / "traffic_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if k != "features"}, indent=2))
