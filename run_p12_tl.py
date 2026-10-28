# -*- coding: utf-8 -*-
"""
run_p12_tl.py — ba phân tích hướng thực hành cho bản tiếng Việt Bài 12 theo góc nhìn
công nghệ hỗ trợ tâm lý trị liệu (dữ liệu tổng hợp, đoàn hệ n = 50 000 của T12).

    .venv/Scripts/python.exe src/run_p12_tl.py          (~5–15 phút, một tiến trình)

Đọc   out/reports/sensitivity_n50k.json, out/reports/p12_tasks.json (T12/T15/T17/T18)
Ghi   out/reports/p12_tl.json, out/reports/p12_tl.txt

TL1  Quy tắc chuyển của Bộ Y tế (QĐ 2254) viết bằng tần suất tự nhiên trên 1 000 trẻ của
     tập kiểm thử: bao nhiêu trẻ chạm ngưỡng, bao nhiêu trong số đó có rối loạn, bao nhiêu
     trẻ có rối loạn bị bỏ sót.
TL2  Hàng đợi khám trong nhóm đã chạm ngưỡng. Mọi trẻ trong nhóm đều được chuyển; điều
     công cụ còn làm được là XẾP THỨ TỰ ai được đánh giá trước. Với điểm hợp nhất (trước
     lưới an toàn) và điểm hệ thống báo ra (sau lưới), đo: tỷ lệ trẻ có rối loạn nằm trong
     10 % / 25 % / 50 % đầu hàng đợi, vị trí trung bình của các em trong hàng đợi, và số
     tuần chờ trung bình ở một cơ sở giả định nhận 100 trẻ, khám 10 trẻ/tuần. Điểm bằng
     nhau được xếp ngẫu nhiên (tính kỳ vọng chính xác, không rút thăm). So sánh cấu hình
     mặc định với các thay đổi một chiều có ý nghĩa ở T17b và với vài trục "vô hại" —
     để quy đổi một hiệu AUC ra số trẻ.
TL3  Gom 18 trục thật theo "ai chịu trách nhiệm" (biểu diễn tri thức lâm sàng / toán tử mờ /
     thành phần học từ dữ liệu / quy định và hợp lệ dữ liệu) và cộng chỉ số Sobol toàn
     phần của mỗi nhóm ở E1, E2, E3 (số liệu có sẵn từ T15, không chạy lại).

Kiểm tính đúng: AUC của cấu hình mặc định tính lại ở đây phải TRÙNG TỪNG BIT với
sensitivity_n50k.json, nếu không chương trình dừng.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SENS_N_RAW", "50000")          # đoàn hệ chính của T12 — đặt TRƯỚC khi import
os.environ.setdefault("SENS_TAG", "_n50k")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import config as cfg  # noqa: E402
import run_sensitivity as rs  # noqa: E402

REP = cfg.REPORTS_DIR
SEED_BOOT_TL = 20260929
N_BOOT = 2000
CAPS = (0.10, 0.25, 0.50)
CLINIC_N, CLINIC_PER_WEEK = 100, 10

# thay đổi một chiều quanh mặc định (nhãn, trục, mức)
VARIANTS = [
    ("SHAPLEY_v1", "SHAPLEY", "v1"),
    ("DEFUZZ_mom", "DEFUZZ", "mom"),
    ("DEFUZZ_bisector", "DEFUZZ", "bisector"),
    ("AGG_owa_doc", "AGG_OP", "owa_doc"),
    ("DHBT_bang_day_du", "DHBT_LAG_TABLE", True),
    ("MF_v1", "MF_SET", "v1"),
    ("T_NORM_min", "T_NORM", "min"),
    ("S_NORM_max", "S_NORM", "max"),
    ("FUSION_luat_troi", "FUSION_WEIGHTS", "luat_troi"),
    ("SAFETY_FLOOR_55", "SAFETY_FLOOR", 55.0),
    ("SAFETY_FLOOR_75", "SAFETY_FLOOR", 75.0),
    ("GIA_LAP_A_1", "GIA_LAP_A", 1),
    # bổ sung theo phản biện MS.22: đủ 6 thay đổi có ý nghĩa ở E3; đặt CUỐI danh sách để hạt giống
    # bootstrap của các thay đổi cũ không đổi
    ("AGG_owa_tuyen_tinh", "AGG_OP", "owa_tuyen_tinh"),
]

GROUPS = {
    "tri_thuc_lam_sang": ["DHBT_LAG_TABLE", "MF_SET", "SHAPLEY", "INTERACTION", "LAG_MAX"],
    "toan_tu_mo": ["T_NORM", "S_NORM", "DEFUZZ", "AGG_OP"],
    "hoc_tu_du_lieu": ["FUSION_WEIGHTS", "FKNN_K", "FKNN_M", "FCM_C", "FCM_M"],
    "quy_dinh_va_hop_le": ["SAFETY_FLOOR", "MISS_THRESHOLD_DOM", "DHBT_ZERO_AS_MISSING", "AGG_POINT_ESTIMATE"],
}


# ------------------------------------------------------------------ hàng đợi có xử lý điểm trùng
def queue_stats(y, s, curve=False):
    """Xếp giảm dần theo điểm; điểm trùng được xếp ngẫu nhiên — trả về kỳ vọng chính xác.

    capture[q]: kỳ vọng tỷ lệ trẻ dương nằm trong q·n vị trí đầu.
    pos_mean: vị trí trung bình (chuẩn hoá 0–1, 0 = đầu hàng) của trẻ dương.
    tie_top: tỷ trọng hồ sơ thuộc nhóm điểm trùng lớn nhất.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n, n1 = len(y), int(y.sum())
    vals, inv, cnt = np.unique(-s, return_inverse=True, return_counts=True)   # thứ tự giảm dần của s
    pos_in = np.bincount(inv, weights=y, minlength=len(vals))
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])                          # vị trí đầu mỗi nhóm
    out = {}
    for q in CAPS:
        k = q * n
        take = np.clip(k - start, 0, cnt)                                       # số chỗ nhóm nhận được
        out[f"capture_{int(q * 100)}"] = float((pos_in * take / cnt).sum() / n1)
    if curve:
        ks = np.linspace(0, 1, 51) * n
        out["curve"] = [float((pos_in * np.clip(k - start, 0, cnt) / cnt).sum() / n1) for k in ks]
    mid = start + (cnt - 1) / 2.0                                              # hạng giữa của nhóm (0-based)
    out["pos_mean"] = float((pos_in * mid).sum() / n1 / (n - 1))
    out["tie_top"] = float(cnt.max() / n)
    out["n_distinct"] = int(len(vals))
    return out


def wait_weeks(pos_mean, n=CLINIC_N, per_week=CLINIC_PER_WEEK):
    """Tuần chờ kỳ vọng (tuần 1 = khám ngay tuần đầu) ở cơ sở nhận n trẻ, khám per_week trẻ/tuần."""
    return 1.0 + pos_mean * (n - 1) / per_week


def paired_boot_queue(y, s_a, s_b, seed):
    """Bootstrap phân tầng (trên nhóm đã chạm ngưỡng) cho hiệu capture_25 và pos_mean: a − b."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d25, dpm = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        qa, qb = queue_stats(y[i], s_a[i]), queue_stats(y[i], s_b[i])
        d25[b] = qa["capture_25"] - qb["capture_25"]
        dpm[b] = qa["pos_mean"] - qb["pos_mean"]
    return {"d_capture_25_se": float(d25.std(ddof=1)),
            "d_capture_25_ci95": [float(np.percentile(d25, 2.5)), float(np.percentile(d25, 97.5))],
            "d_pos_mean_se": float(dpm.std(ddof=1)),
            "d_pos_mean_ci95": [float(np.percentile(dpm, 2.5)), float(np.percentile(dpm, 97.5))]}


def main():
    t0 = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    rs.no_power_throttling()
    ref = json.load(open(os.path.join(REP, "sensitivity_n50k.json"), encoding="utf-8"))
    tasks = json.load(open(os.path.join(REP, "p12_tasks.json"), encoding="utf-8"))
    say("P12 · TL — phân tích hướng thực hành (công nghệ hỗ trợ tâm lý trị liệu)")
    say("dựng đoàn hệ n = 50 000 …")
    rs.init_worker()
    y_all = rs._STATE["y"][rs._STATE["ite"]].astype(int)
    trig = rs._STATE["trig"]
    n, n1 = len(y_all), int(y_all.sum())
    n_t, n1_t = int(trig.sum()), int(y_all[trig].sum())
    say(f"  tập kiểm thử {n} · dương {n1} · chạm ngưỡng BYT {n_t} · dương trong nhóm chạm {n1_t} "
        f"({time.time() - t0:.0f} s)")
    if (n, n1, n_t, n1_t) != (ref["meta"]["n_test"], ref["meta"]["n_pos_test"], ref["meta"]["n_trig"],
                               ref["meta"]["n_pos_trig"]):
        raise SystemExit("đoàn hệ KHÔNG trùng với sensitivity_n50k.json")

    # ---------------------------------------------------------------- TL1
    tl1 = {"n_test": n, "n_pos": n1, "n_trig": n_t, "n_pos_trig": n1_t,
           "n_pos_not_trig": n1 - n1_t, "n_not_trig": n - n_t,
           "per1000_trig": 1000 * n_t / n, "per1000_pos": 1000 * n1 / n,
           "per1000_pos_trig": 1000 * n1_t / n, "per1000_pos_missed": 1000 * (n1 - n1_t) / n,
           "sens_rule": n1_t / n1, "spec_rule": ((n - n_t) - (n1 - n1_t)) / (n - n1),
           "ppv_rule": n1_t / n_t, "npv_rule": ((n - n_t) - (n1 - n1_t)) / (n - n_t),
           "pos_per1000_referred": 1000 * n1_t / n_t}
    say("\nTL1 — quy tắc BYT trên 1 000 trẻ:")
    say(f"  chạm ngưỡng {tl1['per1000_trig']:.1f} · có rối loạn trong nhóm chạm {tl1['per1000_pos_trig']:.1f} · "
        f"bỏ sót {tl1['per1000_pos_missed']:.1f} · độ nhạy {tl1['sens_rule']:.3f} · độ đặc hiệu "
        f"{tl1['spec_rule']:.3f} · GTDĐ {tl1['ppv_rule']:.4f}")

    # ---------------------------------------------------------------- TL2
    base = dict(rs.BASELINE)
    scores = {}
    t1 = time.time()
    raw0, net0 = rs.score_cohort(base)
    scores["mac_dinh"] = (raw0, net0)
    a = (rs.roc_auc(y_all.tolist(), raw0.tolist()), rs.roc_auc(y_all.tolist(), net0.tolist()),
         rs.roc_auc(y_all[trig].tolist(), raw0[trig].tolist()))
    want = (ref["auc_mac_dinh"]["E1"], ref["auc_mac_dinh"]["E2"], ref["auc_mac_dinh"]["E3"])
    if isinstance(want[0], dict):
        want = tuple(w["auc"] if "auc" in w else w for w in want)
    say(f"\n  mặc định: E1 {a[0]!r} · E2 {a[1]!r} · E3 {a[2]!r} ({time.time() - t1:.0f} s)")
    same = all(float(x) == float(w) for x, w in zip(a, want))
    say(f"  so với sensitivity_n50k.json: {'TRÙNG TỪNG BIT' if same else 'KHÁC — dừng'}")
    if not same:
        raise SystemExit(f"AUC mặc định khác: {a} vs {want}")
    for lab, ax, lv in VARIANTS:
        t1 = time.time()
        scores[lab] = rs.score_cohort({**base, ax: lv})
        say(f"  {lab:<18} chấm xong ({time.time() - t1:.0f} s)")

    yt = y_all[trig]
    rows = {}
    rnd = {f"capture_{int(q * 100)}": q for q in CAPS}
    rnd.update({"pos_mean": 0.5})
    for lab, (raw, net) in scores.items():
        for which, s in (("tron", raw), ("bao_ra", net)):
            st = queue_stats(yt, s[trig], curve=True)
            st["auc_trig"] = float(rs.roc_auc(yt.tolist(), s[trig].tolist()))
            st["wait_weeks"] = wait_weeks(st["pos_mean"])
            if which == "bao_ra":
                fl = float(cfg.SAFETY_FLOOR if lab not in ("SAFETY_FLOOR_55", "SAFETY_FLOOR_75")
                           else dict((v[0], v[2]) for v in VARIANTS)[lab])
                st["share_at_floor"] = float(np.mean(np.isclose(s[trig], fl)))
                st["floor"] = fl
            rows[f"{lab}|{which}"] = st
    # phân phối điểm báo ra (mặc định) cho hình: nhóm chạm ngưỡng và nhóm không chạm
    edges = np.arange(0, 101, 2.5)
    hist = {"edges": edges.tolist(),
            "trig": np.histogram(net0[trig], edges)[0].tolist(),
            "not_trig": np.histogram(net0[~trig], edges)[0].tolist(),
            "raw_trig": np.histogram(raw0[trig], edges)[0].tolist(),
            "raw_max_trig": float(raw0[trig].max()), "raw_max_all": float(raw0.max()),
            "n_trig_above_floor_raw": int((raw0[trig] > cfg.SAFETY_FLOOR).sum()),
            "net_min_trig": float(net0[trig].min())}
    # so sánh ghép cặp với mặc định (điểm hợp nhất) và lưới an toàn (báo ra vs hợp nhất)
    cmp_ = {}
    s0 = raw0[trig]
    for i, (lab, (raw, net)) in enumerate(scores.items()):
        if lab == "mac_dinh":
            continue
        pb = paired_boot_queue(yt, raw[trig], s0, SEED_BOOT_TL + i)
        pb["d_capture_25"] = rows[f"{lab}|tron"]["capture_25"] - rows["mac_dinh|tron"]["capture_25"]
        pb["d_pos_mean"] = rows[f"{lab}|tron"]["pos_mean"] - rows["mac_dinh|tron"]["pos_mean"]
        pb["d_auc_trig"] = rows[f"{lab}|tron"]["auc_trig"] - rows["mac_dinh|tron"]["auc_trig"]
        cmp_[lab] = pb
    pb = paired_boot_queue(yt, net0[trig], s0, SEED_BOOT_TL + 99)
    pb["d_capture_25"] = rows["mac_dinh|bao_ra"]["capture_25"] - rows["mac_dinh|tron"]["capture_25"]
    pb["d_pos_mean"] = rows["mac_dinh|bao_ra"]["pos_mean"] - rows["mac_dinh|tron"]["pos_mean"]
    pb["d_auc_trig"] = rows["mac_dinh|bao_ra"]["auc_trig"] - rows["mac_dinh|tron"]["auc_trig"]
    cmp_["luoi_an_toan"] = pb

    say(f"\nTL2 — hàng đợi trong nhóm chạm ngưỡng ({n_t} trẻ, {n1_t} dương = "
        f"{tl1['pos_per1000_referred']:.1f}/1 000); ngẫu nhiên: 10/25/50 % · vị trí 0,500")
    say(f"  {'cấu hình|điểm':<26}{'AUC':>7}{'10%':>7}{'25%':>7}{'50%':>7}{'vị trí':>8}{'tuần':>7}{'trùng lớn nhất':>16}")
    for k, st in rows.items():
        say(f"  {k:<26}{st['auc_trig']:7.3f}{st['capture_10']:7.3f}{st['capture_25']:7.3f}{st['capture_50']:7.3f}"
            f"{st['pos_mean']:8.3f}{st['wait_weeks']:7.2f}{st['tie_top']:16.3f}"
            + (f"  tại sàn {st['share_at_floor']:.3f}" if "share_at_floor" in st else ""))
    say("\n  hiệu so với mặc định (điểm hợp nhất), bootstrap ghép cặp 2 000 lần:")
    for lab, c in cmp_.items():
        say(f"  {lab:<18} ΔAUC {c['d_auc_trig']:+.4f} · Δ bắt 25 % {c['d_capture_25']:+.4f} "
            f"[{c['d_capture_25_ci95'][0]:+.4f}; {c['d_capture_25_ci95'][1]:+.4f}] · Δ vị trí {c['d_pos_mean']:+.4f} "
            f"[{c['d_pos_mean_ci95'][0]:+.4f}; {c['d_pos_mean_ci95'][1]:+.4f}]")

    # ---------------------------------------------------------------- TL3
    sob = tasks["sobol"]["n50k"]
    axes = sob["axes"]
    tl3 = {"groups": GROUPS}
    say("\nTL3 — tổng ST theo nhóm trách nhiệm (n 50 000 chính):")
    for crit in ("E1", "E2", "E3"):
        ST = dict(zip(axes, sob[crit]["ST"]))
        tot = sum(v for k, v in ST.items() if k not in rs.SHAM_AXES)
        floor = tasks["sobol_summary"][crit]["n50k"]["sham_ST_ci_hi"]
        indist = set(tasks["sobol_summary"][crit]["n50k"]["real_axes_indist_from_sham"])
        for g, members in GROUPS.items():
            s = sum(ST[m] for m in members)
            above = [m for m in members if m not in indist]
            tl3[f"{crit}|{g}|sum"] = s
            tl3[f"{crit}|{g}|share"] = s / tot
            tl3[f"{crit}|{g}|above"] = above
            say(f"  {crit} {g:<20} ΣST {s:.3f} ({100 * s / tot:5.1f} %) · trên ngưỡng giả lập: {above}")
        tl3[f"{crit}|floor"] = floor
    assert sorted(sum(GROUPS.values(), [])) == sorted(a for a in axes if a not in rs.SHAM_AXES)

    out = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                    "runtime_s": round(time.time() - t0, 1), "seed_boot": SEED_BOOT_TL, "n_boot": N_BOOT,
                    "caps": list(CAPS), "clinic_n": CLINIC_N, "clinic_per_week": CLINIC_PER_WEEK,
                    "variants": [[l, a_, lv] for l, a_, lv in VARIANTS], "auc_default_bit_identical": same,
                    "n_config_evaluations": 1 + len(VARIANTS)},
           "tl1": tl1, "tl2": {"rows": rows, "cmp": cmp_, "random": rnd, "hist": hist}, "tl3": tl3}
    json.dump(out, open(os.path.join(REP, "p12_tl.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(REP, "p12_tl.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    say(f"\nxong trong {time.time() - t0:.0f} s → out/reports/p12_tl.json")


if __name__ == "__main__":
    main()
