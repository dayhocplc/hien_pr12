# -*- coding: utf-8 -*-
"""
p12_config_audit.py — kiểm toán tham số của một công cụ số bằng TRỤC GIẢ LẬP.

Dành cho nhóm phát triển và cho người thẩm định (kể cả nhà tâm lý lâm sàng tham gia hội đồng
chọn công cụ): nhận một bảng "mỗi dòng một cấu hình đã thử", trả lời câu hỏi "tham số nào
thật sự làm thay đổi chỉ số, tham số nào không phân biệt được với một hằng số không nối vào
đâu cả?". Chỉ cần Python 3 + NumPy.

    python src/p12_config_audit.py cau_hinh.csv --metric auc [--axes A,B,C] [--se 0.012]
           [--sham 2] [--sham-cols X,Y] [--n-perm 5000] [--q 0.05] [--seed 1] [--out bao_cao.json]
    python src/p12_config_audit.py --demo

Đầu vào: CSV, một cột chỉ số (ví dụ AUC trên CÙNG tập đánh giá) và các cột tham số (giá trị
bất kỳ, được coi là mức rời rạc). Khuyến nghị: cấu hình lấy mẫu ngẫu nhiên, cân bằng theo trục.

Công cụ làm năm việc:
  1. Kiểm tra thiết kế: số mức mỗi trục, độ lệch cân bằng (mức ít nhất / nhiều nhất), cột hằng.
  2. Thêm trục giả lập (mặc định 2, gồm một trục 3 mức và một trục 2 mức, gán ngẫu nhiên cân
     bằng) — hoặc dùng các cột giả lập có sẵn (--sham-cols). Trục giả lập không nối vào đâu,
     nên mọi hiệu ứng nó "nhận được" là nhiễu.
  3. Với mỗi trục: η² (tỷ lệ phương sai giữa các mức), giá trị p hoán vị (--n-perm lần),
     hiệu chỉnh Benjamini–Hochberg ở mức FDR --q trên mọi trục KỂ CẢ trục giả lập.
  4. Biên độ biên (trung bình mức tốt nhất − tệ nhất) và, nếu có --se (sai số chuẩn của MỘT
     giá trị chỉ số), quy ra số lần SE.
  5. Kết luận cho từng trục:
       "đáng xem xét"          — qua BH VÀ η² lớn hơn mọi trục giả lập;
       "chưa chắc"             — p hoán vị chưa hiệu chỉnh < --q nhưng không đạt điều kiện trên;
       "không phân biệt được với giả lập".
     Nếu một trục giả lập qua BH, công cụ cảnh báo rằng thủ tục đang không kiểm soát được nhiễu.

Giới hạn (in cùng báo cáo): η² là hiệu ứng chính một chiều, không bắt tương tác; muốn phân rã
đầy đủ dùng chỉ số Sobol trên mô hình thay thế (src/run_p12_tasks.py). Kết luận chỉ đúng trong
vùng mức đã thử và cho đúng chỉ số được đưa vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))


def eta_squared(v, codes):
    grand = v.mean()
    tot = ((v - grand) ** 2).sum()
    if tot <= 0:
        return 0.0
    ss = 0.0
    for lv in np.unique(codes):
        g = v[codes == lv]
        ss += len(g) * (g.mean() - grand) ** 2
    return float(ss / tot)


def perm_p(v, codes, n_perm, rng):
    obs = eta_squared(v, codes)
    c = codes.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(c)
        ge += eta_squared(v, c) >= obs - 1e-15
    return obs, (1 + ge) / (1 + n_perm)


def bh(p, q):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, p[o[i]] * m / (i + 1))
        adj[o[i]] = run
    return adj, adj <= q


def audit(rows, metric, axes=None, sham=2, sham_cols=None, n_perm=5000, q=0.05, seed=1, se=None):
    if not rows:
        raise SystemExit("bảng rỗng")
    cols = list(rows[0].keys())
    if metric not in cols:
        raise SystemExit(f"không có cột chỉ số '{metric}'")
    sham_cols = list(sham_cols or [])
    axes = list(axes) if axes else [c for c in cols if c != metric and c not in sham_cols]
    v, keep = [], []
    for i, r in enumerate(rows):
        try:
            x = float(r[metric])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            v.append(x)
            keep.append(i)
    v = np.asarray(v)
    n = len(v)
    rng = np.random.default_rng(seed)
    warn = []
    if n < len(rows):
        warn.append(f"{len(rows) - n} dòng thiếu chỉ số đã bị bỏ")
    design = {}
    codes = {}
    for a in axes + sham_cols:
        lv = [rows[i][a] for i in keep]
        u, c = np.unique(lv, return_inverse=True)
        codes[a] = c
        cnt = np.bincount(c)
        design[a] = {"n_levels": int(len(u)), "levels": [str(x) for x in u],
                     "balance": float(cnt.min() / cnt.max()) if len(u) > 1 else 1.0}
        if len(u) < 2:
            warn.append(f"trục {a} chỉ có một mức — không đánh giá được")
        elif design[a]["balance"] < 0.5:
            warn.append(f"trục {a} lệch cân bằng (mức ít nhất / nhiều nhất = {design[a]['balance']:.2f})")
    if not sham_cols:
        for j in range(sham):
            k = 3 if j % 2 == 0 else 2
            c = np.resize(np.arange(k), n)
            rng.shuffle(c)
            name = f"GIA_LAP_{j + 1}"
            codes[name] = c
            design[name] = {"n_levels": k, "levels": [str(i) for i in range(k)], "balance": 1.0}
            sham_cols.append(name)
    tested = [a for a in axes if design[a]["n_levels"] > 1] + sham_cols
    res = {}
    for a in tested:
        e, p = perm_p(v, codes[a], n_perm, rng)
        means = [float(v[codes[a] == L].mean()) for L in range(design[a]["n_levels"])]
        res[a] = {"eta2": e, "p_perm": p, "range": max(means) - min(means),
                  "best_level": design[a]["levels"][int(np.argmax(means))],
                  "worst_level": design[a]["levels"][int(np.argmin(means))],
                  "sham": a in sham_cols}
        if se:
            res[a]["range_in_se"] = res[a]["range"] / se
    adj, rej = bh([res[a]["p_perm"] for a in tested], q)
    sham_max = max(res[a]["eta2"] for a in sham_cols)
    for a, pa, r in zip(tested, adj, rej):
        res[a]["p_bh"] = float(pa)
        res[a]["bh"] = bool(r)
        above = res[a]["eta2"] > sham_max
        res[a]["above_sham"] = bool(above) if not res[a]["sham"] else None
        if res[a]["sham"]:
            res[a]["verdict"] = "trục giả lập"
            if r:
                warn.append(f"trục giả lập {a} qua BH — thủ tục KHÔNG kiểm soát được nhiễu; tăng số cấu hình")
        elif r and above:
            res[a]["verdict"] = "đáng xem xét"
        elif res[a]["p_perm"] < q:
            res[a]["verdict"] = "chưa chắc"
        else:
            res[a]["verdict"] = "không phân biệt được với giả lập"
    order = sorted(tested, key=lambda a: -res[a]["eta2"])
    return {"n_configs": n, "metric": metric, "metric_mean": float(v.mean()), "metric_sd": float(v.std(ddof=1)),
            "n_perm": n_perm, "q": q, "seed": seed, "se": se, "sham_cols": sham_cols,
            "sham_eta2_max": sham_max, "design": design, "axes": {a: res[a] for a in order},
            "n_worth": sum(res[a]["verdict"] == "đáng xem xét" for a in tested),
            "n_indist": sum(res[a]["verdict"] == "không phân biệt được với giả lập" for a in tested),
            "n_real": sum(not res[a]["sham"] for a in tested),
            "warnings": warn,
            "limits": ["η² là hiệu ứng chính một chiều; không bắt tương tác (dùng Sobol trên mô hình thay thế)",
                       "kết luận chỉ đúng trong vùng mức đã thử và cho đúng chỉ số được đưa vào",
                       "chỉ số phải đo trên CÙNG một tập đánh giá cho mọi cấu hình"]}


def print_report(r):
    print(f"Chỉ số '{r['metric']}' trên {r['n_configs']} cấu hình: trung bình {r['metric_mean']:.4f}, "
          f"độ lệch chuẩn {r['metric_sd']:.4f}; η² lớn nhất của trục giả lập = {r['sham_eta2_max']:.4f}")
    hdr = f"{'trục':<22}{'mức':>4}{'η²':>8}{'p hoán vị':>11}{'p BH':>8}{'biên độ':>9}"
    hdr += f"{'/SE':>6}" if r["se"] else ""
    print(hdr + "   kết luận")
    for a, x in r["axes"].items():
        line = (f"{a:<22}{r['design'][a]['n_levels']:>4}{x['eta2']:8.4f}{x['p_perm']:11.4f}{x['p_bh']:8.4f}"
                f"{x['range']:9.4f}")
        line += f"{x['range_in_se']:6.2f}" if r["se"] else ""
        print(line + "   " + x["verdict"])
    print(f"→ {r['n_worth']}/{r['n_real']} trục đáng xem xét; {r['n_indist']}/{r['n_real']} không phân biệt được "
          "với giả lập")
    for w in r["warnings"]:
        print("CẢNH BÁO:", w)
    for w in r["limits"]:
        print("Giới hạn:", w)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def demo():
    """Ba kịch bản trên 512 cấu hình của Bài 12 (dữ liệu tổng hợp, tệp demo đi kèm)."""
    path = os.path.join(HERE, "p12_demo_design.csv")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(HERE), "data", "p12_demo_design.csv")
    rows = read_csv(path)
    real = [c for c in rows[0] if c not in ("E1", "E2", "E3", "GIA_LAP_A", "GIA_LAP_B")]
    out = {}
    print("=== A. 512 cấu hình, tiêu chí E3 (xếp thứ tự trong nhóm đã chạm ngưỡng), trục giả lập do công cụ sinh")
    out["A_E3_512"] = audit(rows, "E3", axes=real, se=0.0128)
    print_report(out["A_E3_512"])
    print("\n=== B. Như A nhưng dùng hai trục giả lập nằm sẵn trong thiết kế (GIA_LAP_A, GIA_LAP_B), tiêu chí E1")
    out["B_E1_512_sham_thiet_ke"] = audit(rows, "E1", axes=real, sham_cols=["GIA_LAP_A", "GIA_LAP_B"], se=0.0119)
    print_report(out["B_E1_512_sham_thiet_ke"])
    print("\n=== C. Chỉ 64 cấu hình đầu (thiết kế nhỏ), tiêu chí E3")
    out["C_E3_64"] = audit(rows[:64], "E3", axes=real, se=0.0128)
    print_report(out["C_E3_64"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--metric")
    ap.add_argument("--axes")
    ap.add_argument("--sham", type=int, default=2)
    ap.add_argument("--sham-cols")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--se", type=float)
    ap.add_argument("--out")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        r = demo()
        dst = a.out or os.path.join(os.path.dirname(HERE), "out", "reports", "p12_audit_demo.json")
    else:
        if not a.csv or not a.metric:
            ap.error("cần tệp CSV và --metric (hoặc --demo)")
        r = audit(read_csv(a.csv), a.metric, axes=a.axes.split(",") if a.axes else None, sham=a.sham,
                  sham_cols=a.sham_cols.split(",") if a.sham_cols else None, n_perm=a.n_perm, q=a.q, seed=a.seed,
                  se=a.se)
        print_report(r)
        dst = a.out
    if dst:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        json.dump(r, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nđã ghi {dst}")


if __name__ == "__main__":
    main()
