# -*- coding: utf-8 -*-
"""
run_p12_tasks.py — P12 · T12 + T15 + T17 + T18 (việc còn phải làm của bài P12).

    SENS_N_RAW=50000 SENS_TAG=_n50k   .venv/Scripts/python.exe src/run_sensitivity.py
    SENS_N_RAW=50000 SENS_TAG=_n50k_rep SENS_CORE_ONLY=1 \
        SENS_SEED_COHORT=20260927 SENS_SEED_DESIGN=20260928 \
        .venv/Scripts/python.exe src/run_sensitivity.py
    .venv/Scripts/python.exe src/run_p12_tasks.py

Đầu vào : out/reports/sensitivity_n50k.json      (T12 — lần quét chính, n = 50 000)
          out/reports/sensitivity_n50k_rep.json  (lần lặp ĐỘC LẬP: đoàn hệ + thiết kế mới)
          out/reports/sensitivity.json           (lần quét cũ n = 20 000, 29/08)
Đầu ra  : out/reports/p12_tasks.json, p12_tasks.txt, p12_config_log.json

Bốn việc tệp này bổ sung so với run_sensitivity.py
--------------------------------------------------
T12  Đánh giá lại trên n = 50 000 (đoàn hệ ≈ 30 700, tập kiểm thử ≈ 15 300).
T15  (a) Ước lượng Sobol QUY TÂM (Saltelli 2010 cho S1, Jansen cho ST) — bản cũ
     không quy tâm nên S1 có thể âm hoặc vượt ST; (b) khoảng tin cậy bằng
     bootstrap Bayes MÔ HÌNH THAY THẾ (trọng số Dirichlet trên 512 cấu hình); (c) độ
     ổn định xếp hạng giữa ba lần quét (n 20 000 · n 50 000 · n 50 000 độc lập).
T17  (a) BH trong từng tiêu chí (như cũ) và BH GỘP trên cả 3 × 20 phép kiểm;
     (b) họ phép kiểm MỚI: hiệu AUC GHÉP CẶP trên cùng tập kiểm thử cho mọi mức
     của mọi trục quanh cấu hình mặc định (bootstrap ghép cặp), hiệu chỉnh BH.
     Lý do: "biên độ < 1 SE" trong bản nháp so hiệu của hai cấu hình với SE
     của MỘT AUC — thang đó đúng cho so sánh giữa hai mẫu độc lập, nhưng quá
     rộng cho so sánh ghép cặp trên cùng dữ liệu (hai điểm số tương quan mạnh).
     Bài phải báo cáo cả hai thang.
     (c) khai báo tổng số cấu hình đã đánh giá (§11.5).
T18  Nhật ký cấu hình (hạt giống, phiên bản, SHA-256) và kiểm tra tái lập TỪNG
     BIT: chấm lại một số cấu hình của thiết kế trong tiến trình chính và so
     với giá trị do tiến trình con tính.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("SENS_N_RAW", "50000")      # T12 — phải đặt TRƯỚC khi import
os.environ.setdefault("SENS_TAG", "_n50k")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

# sklearn 1.9 cảnh báo mỗi lần rừng dự báo song song từ luồng — vô hại, nhưng
# lặp hàng triệu lần làm phình nhật ký (đã gặp: 64 MB).
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

import config as cfg
import run_sensitivity as rs

# Ảnh chụp giá trị MẶC ĐỊNH của config lúc nạp mô-đun: rs._apply() ghi đè các
# hằng số này trong tiến trình chính (lát cắt, T18), nên không được đọc lại sau.
_CFG_KEYS = ("T_NORM", "S_NORM", "DEFUZZ", "SHAPLEY", "INTERACTION", "OWA_WEIGHTS",
             "DHBT_USE_LAG_TABLE", "LAG_MAX", "MISS_THRESHOLD_DOM", "DHBT_ZERO_AS_MISSING",
             "AGG_POINT_ESTIMATE", "SAFETY_FLOOR", "FUSION_WEIGHTS", "FKNN_K", "FKNN_M",
             "FCM_C", "FCM_M")


def _jsonable(v):
    if isinstance(v, dict):
        return {("|".join(k) if isinstance(k, tuple) else str(k)): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


CFG_DEFAULTS = {k: _jsonable(getattr(cfg, k)) for k in _CFG_KEYS if hasattr(cfg, k)}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = cfg.REPORTS_DIR
SEED_PAIRED = 20260927
SEED_SURR_BOOT = 20260928
N_BOOT_PAIRED = 2000
N_SURR_BOOT = 40
N_MC_CI = 8000
CRITERIA = ("E1", "E2", "E3")
REAL_AXES = [a for a in rs.AXIS_NAMES if a not in rs.SHAM_AXES]


def load(tag):
    p = os.path.join(REP, "sensitivity{}.json".format(tag))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def codes_of(d):
    """Dựng lại ma trận mã mức từ nhật ký cấu hình (chuỗi) của một lần quét."""
    ax = d["axes"]
    names = list(ax)
    C = np.array([[ax[a].index(row[a]) for a in names]
                  for row in d["nhat_ky_cau_hinh"]], dtype=int)
    return names, C


# =====================================================================
# T15 — Sobol quy tâm + bootstrap mô hình thay thế
# =====================================================================
def _encode(C, n_lv):
    cols = []
    for j in range(C.shape[1]):
        for lv in range(n_lv[j]):
            cols.append((C[:, j] == lv).astype(float))
    return np.column_stack(cols)


def sobol_centered(rf, n_lv, n_mc, rng):
    d = len(n_lv)
    A = np.column_stack([rng.integers(0, n_lv[j], n_mc) for j in range(d)])
    B = np.column_stack([rng.integers(0, n_lv[j], n_mc) for j in range(d)])
    fA = rf.predict(_encode(A, n_lv))
    fB = rf.predict(_encode(B, n_lv))
    mu = 0.5 * (fA.mean() + fB.mean())
    fA, fB = fA - mu, fB - mu
    var = float(np.var(np.concatenate([fA, fB])))
    S1, ST = np.zeros(d), np.zeros(d)
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = rf.predict(_encode(AB, n_lv)) - mu
        S1[j] = np.mean(fB * (fAB - fA)) / var            # Saltelli 2010
        ST[j] = 0.5 * np.mean((fA - fAB) ** 2) / var       # Jansen 1999
    return S1, ST, var


def surrogate_sobol(C, y, n_lv, seed):
    from sklearn.ensemble import RandomForestRegressor
    X = _encode(C, n_lv)
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                               oob_score=True, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    S1, ST, var = sobol_centered(rf, n_lv, 40_000, np.random.default_rng(seed))
    # Bootstrap BAYES của mô hình thay thế (trọng số Dirichlet trên 512 cấu
    # hình, khớp lại rừng, tính lại Sobol). Bootstrap đa thức lặp lại hàng —
    # rừng với min_samples_leaf = 2 khớp vào các bản sao và co các chỉ số nhỏ
    # về 0 (thử trên n 20 000: KTC của MF_SET [0,012; 0,025] không chứa ước
    # lượng điểm 0,040). Trọng số liên tục không tạo bản sao nên không mắc lỗi đó.
    rng = np.random.default_rng(SEED_SURR_BOOT)
    bS1, bST = [], []
    for b in range(N_SURR_BOOT):
        w = rng.dirichlet(np.ones(len(y))) * len(y)
        rb = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=seed + 1 + b, n_jobs=-1)
        rb.fit(X, y, sample_weight=w)
        s1, st, _ = sobol_centered(rb, n_lv, N_MC_CI,
                                   np.random.default_rng(seed + 101 + b))
        bS1.append(s1)
        bST.append(st)
    bS1, bST = np.array(bS1), np.array(bST)

    def centred(point, boot):
        # Khớp lại rừng trên dữ liệu có trọng số làm giảm cỡ mẫu hiệu dụng nên
        # co nhẹ các chỉ số nhỏ về 0; KTC được đặt quanh ước lượng từ dữ liệu
        # đầy đủ: điểm + (phân vị − trung vị bootstrap).
        q = np.percentile(boot, [2.5, 50, 97.5], axis=0)
        return np.column_stack([point + q[0] - q[1], point + q[2] - q[1]]).tolist()

    return {"oob_r2": float(rf.oob_score_), "var": var,
            "S1": S1.tolist(), "ST": ST.tolist(),
            "S1_ci": centred(S1, bS1), "ST_ci": centred(ST, bST),
            "S1_boot_median": np.median(bS1, axis=0).tolist(),
            "ST_boot_median": np.median(bST, axis=0).tolist(),
            "ST_ci_raw_percentile": np.percentile(bST, [2.5, 97.5], axis=0).T.tolist()}


def sobol_run(run_key, d):
    """Sobol cho cả ba tiêu chí của một lần quét, có bộ đệm theo SHA-256 của
    AUC từng cấu hình (tính lại khi kết quả quét đổi)."""
    names, C = codes_of(d)
    key = hashlib.sha256(json.dumps([d["auc_tung_cau_hinh"], d["nhat_ky_cau_hinh"],
                                     N_SURR_BOOT, N_MC_CI, "centred-v2"]).encode()).hexdigest()[:16]
    cp = os.path.join(REP, "_p12", "sobol_{}_{}.json".format(run_key, key))
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    n_lv = [len(d["axes"][a]) for a in names]
    out = {"axes": names}
    for k, crit in enumerate(CRITERIA):
        y = np.array(d["auc_tung_cau_hinh"][crit], dtype=float)
        ok = np.isfinite(y)
        r = surrogate_sobol(C[ok], y[ok], n_lv, rs.SEED_SURROGATE + 10 * k)
        r["eta2"] = [row["eta2"] for row in d["phan_tich"][crit]["truc"]]
        r["n"] = int(ok.sum())
        out[crit] = r
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


# =====================================================================
# T17 — hiệu AUC ghép cặp quanh cấu hình mặc định
# =====================================================================
def _job_scores(levels):
    raw, net = rs.score_cohort(levels)
    return raw, net


def _auc_fast(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def paired_boot(y, s_a, s_b, n_boot, seed):
    """SE bootstrap phân tầng của AUC(s_a) − AUC(s_b) trên CÙNG chỉ số mẫu."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        d[b] = _auc_fast(y[i], s_a[i]) - _auc_fast(y[i], s_b[i])
    delta = _auc_fast(y, s_a) - _auc_fast(y, s_b)
    se = float(d.std(ddof=1))
    # p hai phía theo xấp xỉ chuẩn trên SE bootstrap (Wald)
    from scipy.stats import norm
    z = delta / se if se > 0 else 0.0
    return {"delta": float(delta), "se": se, "z": float(z),
            "p": float(2 * norm.sf(abs(z))) if se > 0 else 1.0,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def bh(p, q=0.05):
    return rs.benjamini_hochberg(list(p), q=q)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rs.no_power_throttling()
    t_start = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    prim = load("_n50k")
    repl = load("_n50k_rep")
    old = load("")
    res = {"meta": {"started": time.strftime("%Y-%m-%d %H:%M:%S")}}

    say("=" * 96)
    say("P12 · T12 + T15 + T17 + T18")
    say("=" * 96)
    for name, d in (("n 20 000 (29/08)", old), ("n 50 000 chính", prim),
                    ("n 50 000 lặp độc lập", repl)):
        m = d["meta"]
        say("  {:<22} đoàn hệ {:>6} · kiểm thử {:>6} · dương {:>4} · chạm BYT {:.4f}"
            " · hạt giống đoàn hệ {} · thiết kế {}".format(
                name, m["n_cohort"], m["n_test"],
                m.get("n_pos_test", round(m["ty_le_duong"] * m["n_test"])),
                m["ty_le_cham_byt"], m["seeds"]["cohort"], m["seeds"]["design"]))
    res["runs"] = {k: {"meta": d["meta"], "auc_default": d["auc_mac_dinh"],
                       "noise": {kk: vv for kk, vv in d["nguong_nhieu"].items()
                                 if kk != "lap_lai"}}
                   for k, d in (("n20k", old), ("n50k", prim), ("n50k_rep", repl))}

    # -----------------------------------------------------------------
    # T15 — Sobol quy tâm + khoảng tin cậy + độ ổn định xếp hạng
    # -----------------------------------------------------------------
    say()
    say("-" * 96)
    say("T15 — SOBOL QUY TÂM, BOOTSTRAP MÔ HÌNH THAY THẾ, ỔN ĐỊNH XẾP HẠNG")
    say("-" * 96)
    sob = {}
    for run_key, d in (("n50k", prim), ("n50k_rep", repl), ("n20k", old)):
        sob[run_key] = sobol_run(run_key, d)
        say("  {} xong ({:.0f} s)".format(run_key, time.time() - t_start))
    res["sobol"] = sob

    def table(run_key, crit):
        r = sob[run_key][crit]
        names = sob[run_key]["axes"]
        ST = np.array(r["ST"])
        real = np.array([a not in rs.SHAM_AXES for a in names])
        order = np.argsort(-ST)
        top3 = [names[i] for i in order if real[i]][:3]
        share = float(np.sort(ST[real])[-3:].sum() / ST[real].sum())
        sham_max = float(ST[~real].max())
        sham_ci_hi = float(max(r["ST_ci"][i][1] for i in np.flatnonzero(~real)))
        below = [names[i] for i in np.flatnonzero(real)
                 if r["ST_ci"][i][0] <= sham_ci_hi]
        return top3, share, sham_max, sham_ci_hi, below

    summary = {}
    for crit in CRITERIA:
        say()
        say("  Tiêu chí {} — n 50 000 chính (R² OOB {:.3f}):".format(
            crit, sob["n50k"][crit]["oob_r2"]))
        say("  {:<22}{:>8}{:>18}{:>8}{:>18}{:>8}".format(
            "trục", "ST", "KTC95 ST", "S1", "KTC95 S1", "η²"))
        r = sob["n50k"][crit]
        names = sob["n50k"]["axes"]
        for i in np.argsort(-np.array(r["ST"])):
            say("  {:<22}{:>8.3f}   [{:>6.3f};{:>6.3f}]{:>8.3f}   [{:>6.3f};{:>6.3f}]{:>8.3f}"
                .format(names[i], r["ST"][i], *r["ST_ci"][i], r["S1"][i],
                        *r["S1_ci"][i], r["eta2"][i]))
        summary[crit] = {}
        for run_key in ("n50k", "n50k_rep", "n20k"):
            top3, share, smax, shi, below = table(run_key, crit)
            summary[crit][run_key] = {"top3": top3, "top3_share": share,
                                      "sham_ST_max": smax,
                                      "sham_ST_ci_hi": shi,
                                      "real_axes_indist_from_sham": below,
                                      "oob_r2": sob[run_key][crit]["oob_r2"]}
            say("  [{}] top-3 {} · chiếm {:.1%} ST của trục thật · trần KTC giả lập {:.4f} · "
                "{} trục thật có KTC dưới chạm trần đó".format(
                    run_key, top3, share, shi, len(below)))
        # ổn định xếp hạng
        stab = {}
        for a_key, b_key in (("n50k", "n50k_rep"), ("n50k", "n20k"),
                             ("n50k_rep", "n20k")):
            sa = np.array(sob[a_key][crit]["ST"])
            sb = np.array(sob[b_key][crit]["ST"])
            rho = spearmanr(sa, sb).statistic
            tau = kendalltau(sa, sb).statistic
            stab["{}~{}".format(a_key, b_key)] = {"spearman": float(rho),
                                                  "kendall": float(tau)}
            say("  ổn định ST {} ~ {}: Spearman {:.3f} · Kendall {:.3f}"
                .format(a_key, b_key, rho, tau))
        summary[crit]["stability"] = stab
    res["sobol_summary"] = summary

    # -----------------------------------------------------------------
    # T17 (a) — BH gộp trên 60 phép kiểm hoán vị
    # -----------------------------------------------------------------
    say()
    say("-" * 96)
    say("T17a — BH TRONG TỪNG TIÊU CHÍ VÀ BH GỘP 3 × 20 PHÉP KIỂM (n 50 000 chính)")
    say("-" * 96)
    allp, lab = [], []
    for crit in CRITERIA:
        for row in prim["phan_tich"][crit]["truc"]:
            allp.append(row["p"])
            lab.append((crit, row["truc"]))
    g = bh(allp)
    joint = {}
    for (crit, a), padj, rej in zip(lab, g["p_hieu_chinh"], g["bac_bo"]):
        joint.setdefault(crit, {})[a] = {"p_BH_joint": padj, "sig": bool(rej)}
    for crit in CRITERIA:
        within = [r["truc"] for r in prim["phan_tich"][crit]["truc"]
                  if r["y_nghia_BH"]]
        jn = [a for a, v in joint[crit].items() if v["sig"]]
        say("  {}: trong tiêu chí {} trục {} · gộp {} trục {}".format(
            crit, len(within), within, len(jn), jn))
    res["bh_joint"] = {"n_tests": len(allp), "k_rejected": g["k_bac_bo"],
                       "by_criterion": joint}
    min_p = 1.0 / (1 + rs.N_PERM)
    say("  (giá trị p hoán vị nhỏ nhất đạt được = 1/(1+{}) = {:.5f})"
        .format(rs.N_PERM, min_p))

    # -----------------------------------------------------------------
    # T17 (b) — hiệu AUC ghép cặp quanh cấu hình mặc định (mọi trục, mọi mức)
    # -----------------------------------------------------------------
    say()
    say("-" * 96)
    say("T17b — HIỆU AUC GHÉP CẶP, CÙNG TẬP KIỂM THỬ, QUANH CẤU HÌNH MẶC ĐỊNH")
    say("-" * 96)
    jobs = [("baseline", None, dict(rs.BASELINE))]
    for a in rs.AXIS_NAMES:
        for lv in rs.AXES[a]:
            if lv == rs.BASELINE[a]:
                continue
            jobs.append((a, lv, {**rs.BASELINE, a: lv}))
    t0 = time.time()
    n_workers = int(os.environ.get("SENS_WORKERS", "0")) or max(1, min(10, (os.cpu_count() or 4) - 2))
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=rs.init_worker) as ex:
        outs = list(ex.map(_job_scores, [j[2] for j in jobs]))
    say("  {} cấu hình lát cắt chấm xong trong {:.0f} s".format(
        len(jobs), time.time() - t0))

    rs.init_worker()                                   # đoàn hệ chính, tiến trình này
    y_te = rs._STATE["y"][rs._STATE["ite"]].astype(int)
    trig = rs._STATE["trig"]
    raw0, net0 = outs[0]
    se1 = prim["nguong_nhieu"]["bootstrap_E1"]["se"]
    se3 = prim["nguong_nhieu"]["bootstrap_E3"]["se"]

    # hiệu ứng của lưới an toàn, ghép cặp
    sn = paired_boot(y_te, raw0, net0, N_BOOT_PAIRED, SEED_PAIRED)
    say("  Lưới an toàn (E1 − E2, mặc định): Δ = {:.4f} · SE ghép cặp {:.4f} · "
        "= {:.2f} SE đơn · z = {:.1f}".format(sn["delta"], sn["se"],
                                             sn["delta"] / se1, sn["z"]))
    res["safety_net_paired"] = {**sn, "per_single_se": sn["delta"] / se1}

    comps = []
    for (a, lv, _), (raw, net) in zip(jobs[1:], outs[1:]):
        for crit, yy, sa, sb, se_single in (
                ("E1", y_te, raw, raw0, se1),
                ("E3", y_te[trig], raw[trig], raw0[trig], se3)):
            r = paired_boot(yy, sa, sb, N_BOOT_PAIRED,
                            SEED_PAIRED + len(comps) + 1)
            comps.append({"axis": a, "level": str(lv), "criterion": crit,
                          "sham": a in rs.SHAM_AXES,
                          "delta_single_se": r["delta"] / se_single, **r})
    for crit in ("E1", "E3"):
        idx = [i for i, c in enumerate(comps) if c["criterion"] == crit]
        b = bh([comps[i]["p"] for i in idx])
        for i, padj, rej in zip(idx, b["p_hieu_chinh"], b["bac_bo"]):
            comps[i]["p_BH"] = padj
            comps[i]["sig_BH"] = bool(rej)
    res["paired_slices"] = comps

    say("  {:<22}{:<16}{:>4}{:>9}{:>9}{:>8}{:>8}{:>9}".format(
        "trục", "mức", "tc", "Δ AUC", "SE ghép", "z", "/SE đơn", "p BH"))
    for c in comps:
        say("  {:<22}{:<16}{:>4}{:>+9.4f}{:>9.4f}{:>8.2f}{:>8.2f}{:>9.4f}{}".format(
            c["axis"], c["level"][:15], c["criterion"], c["delta"], c["se"],
            c["z"], c["delta_single_se"], c["p_BH"], " *" if c["sig_BH"] else ""))
    for crit in ("E1", "E3"):
        cc = [c for c in comps if c["criterion"] == crit]
        n_sig = sum(c["sig_BH"] for c in cc)
        n_sig_sham = sum(c["sig_BH"] for c in cc if c["sham"])
        max_single = max(abs(c["delta_single_se"]) for c in cc)
        ratio = np.median([c["se"] / (se1 if crit == "E1" else se3) for c in cc
                           if not c["sham"]])
        say("  {}: {}/{} so sánh có ý nghĩa sau BH (trục giả lập: {}) · "
            "|Δ| lớn nhất = {:.2f} SE đơn · trung vị SE ghép / SE đơn = {:.2f}"
            .format(crit, n_sig, len(cc), n_sig_sham, max_single, ratio))
        res.setdefault("paired_summary", {})[crit] = {
            "n": len(cc), "n_sig": int(n_sig), "n_sig_sham": int(n_sig_sham),
            "max_abs_delta_single_se": float(max_single),
            "median_paired_to_single_se": float(ratio)}

    # -----------------------------------------------------------------
    # T18 — tái lập từng bit
    # -----------------------------------------------------------------
    say()
    say("-" * 96)
    say("T18 — TÁI LẬP TỪNG BIT (chấm lại cấu hình thiết kế trong tiến trình chính)")
    say("-" * 96)
    names, C = codes_of(prim)
    pick = [0, 101, 257, 511]
    repro = []
    for i in pick:
        lv = {a: rs.AXES[a][C[i, j]] for j, a in enumerate(names)}
        e = rs.eval_config(lv)
        stored = [prim["auc_tung_cau_hinh"][k][i] for k in CRITERIA]
        same = all((x is None and s is None) or x == s for x, s in zip(e, stored))
        repro.append({"config": i, "recomputed": list(e), "stored": stored,
                      "bit_identical": bool(same)})
        say("  cấu hình {:>3}: {} → {}".format(
            i, ["{:.15f}".format(x) for x in e],
            "TRÙNG TỪNG BIT" if same else "KHÁC"))
    res["reproducibility"] = repro

    # -----------------------------------------------------------------
    # T17 (c) + T18 — nhật ký cấu hình
    # -----------------------------------------------------------------
    n_extra_prim = (1 + len(prim["nguong_nhieu"]["lap_lai"])
                    + sum(len(v) for v in prim["lat_cat"].values())
                    + len(prim["do_phu"]) + len(prim["rang_buoc_hang_so"]))
    counts = {
        "design_primary_n50k": prim["meta"]["n_config"],
        "design_replicate_n50k": repl["meta"]["n_config"],
        "design_original_n20k": old["meta"]["n_config"],
        "auxiliary_primary (baseline, 5 replica seeds, slices, coverage, coupling)":
            n_extra_prim,
        "auxiliary_replicate (baseline)": 1,
        "paired_slices_T17b": len(jobs),
        "reproducibility_T18": len(pick),
    }
    counts["total_configuration_evaluations"] = sum(counts.values())
    files = ["src/run_sensitivity.py", "src/run_p12_tasks.py", "src/config.py",
             "src/ts_fuzzy.py", "src/ts_aggregate.py", "src/ts_fusion.py",
             "src/ts_fknn.py", "src/ts_fcm.py", "src/ts_features.py",
             "src/ts_simulate.py", "src/ts_mf.py", "src/ts_dataset.py",
             "out/reports/sensitivity.json", "out/reports/sensitivity_n50k.json",
             "out/reports/sensitivity_n50k_rep.json"]
    import scipy
    import sklearn
    log = {
        "task": "P12 · T12 + T15 + T17 + T18",
        "run_started": res["meta"]["started"],
        "runtime_s": None,
        "argv": sys.argv,
        "commands": [
            "SENS_N_RAW=50000 SENS_TAG=_n50k python src/run_sensitivity.py",
            "SENS_N_RAW=50000 SENS_TAG=_n50k_rep SENS_CORE_ONLY=1 "
            "SENS_SEED_COHORT=20260927 SENS_SEED_DESIGN=20260928 "
            "python src/run_sensitivity.py",
            "python src/run_p12_tasks.py"],
        "seeds": {"primary": prim["meta"]["seeds"],
                  "replicate": repl["meta"]["seeds"],
                  "original_n20k": old["meta"]["seeds"],
                  "paired_bootstrap": SEED_PAIRED,
                  "surrogate_bootstrap": SEED_SURR_BOOT},
        "design": {"axes": {a: [str(x) for x in v] for a, v in rs.AXES.items()},
                   "baseline": {k: str(v) for k, v in rs.BASELINE.items()},
                   "n_grid_full": rs.N_GRID_FULL, "n_perm": rs.N_PERM,
                   "n_boot": rs.N_BOOT, "n_boot_paired": N_BOOT_PAIRED,
                   "n_mc_sobol": 40000, "n_surrogate_boot": N_SURR_BOOT,
                   "n_mc_ci": N_MC_CI, "fdr_q": rs.FDR_Q},
        "hypothesis_families": {
            "F1_permutation_within_criterion": "20 axes x 3 criteria, BH per criterion",
            "F1_joint": "60 tests, BH jointly (sensitivity)",
            "F2_paired_slices": "{} comparisons per criterion (E1, E3), BH per criterion"
                                .format(len(jobs) - 1)},
        "configuration_counts": counts,
        "config_defaults": CFG_DEFAULTS,
        "sha256": {f: sha(os.path.join(PROJECT_DIR, f)) for f in files
                   if os.path.exists(os.path.join(PROJECT_DIR, f))},
        "versions": dict(python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__,
                         platform=platform.platform()),
        "reproducibility": repro,
    }
    say()
    say("T17c — số lần đánh giá cấu hình (khai báo §11.5):")
    for k, v in counts.items():
        say("  {:<70}{:>6}".format(k, v))

    log["runtime_s"] = round(time.time() - t_start, 1)
    res["config_counts"] = counts
    res["meta"]["runtime_s"] = log["runtime_s"]
    with open(os.path.join(REP, "p12_config_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(REP, "p12_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(REP, "p12_tasks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say("Đã ghi out/reports/p12_tasks.{{json,txt}}, p12_config_log.json "
        "({:.0f} s)".format(time.time() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
