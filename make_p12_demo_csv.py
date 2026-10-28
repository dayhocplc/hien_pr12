# -*- coding: utf-8 -*-
"""Xuất 512 cấu hình của thiết kế chính (n 50 000) + ba AUC ra data/p12_demo_design.csv — đầu vào demo của
p12_config_audit.py.   .venv/Scripts/python.exe src/make_p12_demo_csv.py"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = json.load(open(os.path.join(ROOT, "out", "reports", "sensitivity_n50k.json"), encoding="utf-8"))
log, auc = s["nhat_ky_cau_hinh"], s["auc_tung_cau_hinh"]
axes = list(s["axes"])
dst = os.path.join(ROOT, "data", "p12_demo_design.csv")
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(axes + ["E1", "E2", "E3"])
    for i, c in enumerate(log):
        w.writerow([c[a] for a in axes] + [repr(float(auc[k][i])) for k in ("E1", "E2", "E3")])
print(f"{len(log)} cấu hình → {dst}")
