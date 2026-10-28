# -*- coding: utf-8 -*-
"""
run_p12_r1.py — phân tích bổ sung cho bản sửa Bài 12 theo phiếu phản biện MS.22 (Hội thảo
"Ứng dụng công nghệ trong đánh giá và can thiệp các rối loạn phát triển thần kinh", 2026).

    .venv/Scripts/python.exe src/run_p12_r1.py --part data        (~5 phút)
    .venv/Scripts/python.exe src/run_p12_r1.py --part floor --gen "prev=0.01"   (gọi nội bộ)
    .venv/Scripts/python.exe src/run_p12_r1.py --part scen        (cần sensitivity_r1_*.json)

Ghi out/reports/p12_r1.json (+ .txt). Các phần:

R1  Phân rã tỷ lệ chạm ngưỡng 86,3 % trên tập kiểm thử chính: chạm qua tiêu chí lĩnh vực chính
    (≥ 2/3 câu "Không"), qua tiêu chí dấu hiệu bất thường (≥ 1 dấu hiệu), cả hai; theo nhóm tuổi;
    tỷ lệ lĩnh vực kích hoạt khi chỉ có ≤ 2 câu được trả lời.            (phản biện: yêu cầu 4)
R2  Đối chiếu dữ liệu mô phỏng với đoàn hệ lâm sàng Ba Lan (Q-CHAT-25, n = 252, nhãn chẩn đoán lâm sàng tại cộng đồng;
    Mendeley doi:10.17632/tmpkt2mfkg.2) trên bốn lĩnh vực chung, nhóm tuổi 18–24 tháng. Tiêu chí đặt trước:
      C1 cùng chiều: tỷ lệ câu nguy cơ ở nhóm dương > nhóm âm, ở cả 4 lĩnh vực, ở cả hai nguồn;
      C2 cùng thứ bậc: tương quan hạng (Spearman) của AUC theo lĩnh vực giữa hai nguồn;
      C3 tỷ lệ câu nguy cơ trung bình ở trẻ không mắc cùng bậc độ lớn (tỷ số trong [0,5; 2]).
    Hai công cụ khác nhau (20 câu Có/Không so với 25 câu thang 0–4) nên đây là kiểm tra tính
    hợp lý định tính, không phải kiểm định tương đương.                  (phản biện: yêu cầu 3)
R3  Ảnh hưởng của việc chỉ giữ hồ sơ chấm đủ 6 lĩnh vực (61,4 %): so sánh hồ sơ bị loại và
    được giữ; chấm lại cấu hình mặc định trên đoàn hệ nới (ngưỡng bỏ trống 0,5 → 94,1 %).
                                                                          (phản biện: khuyến nghị 12)
R4  Ba kịch bản giả định của bộ sinh (tỷ lệ mắc 1 %, 5 %; chạm ngưỡng thấp) — đọc
    sensitivity_r1_*.json, tính Sobol quy tâm, ngưỡng giả lập, BH, mất AUC do lưới, tỷ lệ trẻ
    được chuyển dồn về sàn.                                               (phản biện: yêu cầu 4)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("SENS_N_RAW", "50000")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import config as cfg  # noqa: E402
import run_sensitivity as rs  # noqa: E402

REP = cfg.REPORTS_DIR
OUT = os.path.join(REP, "p12_r1.json")
SCEN = [("prev01", "prev=0.01", "Tỷ lệ mắc 1 %"),
        ("prev05", "prev=0.05", "Tỷ lệ mắc 5 %"),
        ("lowtrig", "theta=2;dhbt=-4;noise=0.01;noiseabs=0.03", "Chạm ngưỡng thấp")]
AGE_BANDS = [(180, 365), (365, 545), (545, 730), (730, 913), (913, 1096)]
SHARED = ["CNXH", "GTNN", "BCHOC", "DHBT"]


def load_out():
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def save_out(d):
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def auc(y, s):
    return float(rs.roc_auc(list(map(int, y)), list(map(float, s))))


# ------------------------------------------------------------------ R1
def part_trigger(say):
    from ts_fusion import byt_threshold_met, domain_risk_counts
    q = rs._STATE["questions"]
    recs = rs._STATE["cohort"]
    ite = rs._STATE["ite"]
    y = rs._STATE["y"][ite].astype(int)
    main, dh, small, age = [], [], [], []
    for i in ite:
        r = recs[i]
        ok, reasons = byt_threshold_met(r, q)
        m = any("Dấu hiệu" not in x for x in reasons)
        d = any("Dấu hiệu" in x for x in reasons)
        main.append(m)
        dh.append(d)
        age.append(r.age_days)
        c = domain_risk_counts(r, q)
        small.append(m and all(c[k][1] <= 2 for k in cfg.MAIN_DOMAINS
                               if k in c and c[k][1] and c[k][0] / c[k][1] >= cfg.BYT_DOMAIN_RATIO))
    main, dh, small, age = map(np.array, (main, dh, small, age))
    trig = main | dh
    if not np.array_equal(trig, rs._STATE["trig"]):
        raise SystemExit("phân rã không khớp cờ chạm ngưỡng")
    neg = y == 0
    out = {"trig": float(trig.mean()), "main": float(main.mean()), "dhbt": float(dh.mean()),
           "both": float((main & dh).mean()), "main_only": float((main & ~dh).mean()),
           "dhbt_only": float((dh & ~main).mean()),
           "trig_neg": float(trig[neg].mean()), "trig_pos": float(trig[~neg].mean()),
           "main_neg": float(main[neg].mean()), "dhbt_neg": float(dh[neg].mean()),
           "main_via_small_domain": float(small[main].mean()),
           "by_age": []}
    for lo, hi in AGE_BANDS:
        mm = (age >= lo) & (age < hi)
        out["by_age"].append({"lo": lo, "hi": hi, "n": int(mm.sum()),
                              "trig_neg": float(trig[mm & neg].mean()), "main_neg": float(main[mm & neg].mean()),
                              "dhbt_neg": float(dh[mm & neg].mean())})
    # tỷ lệ "có dấu hiệu" của từng câu DHBT ở trẻ không mắc
    from ts_fusion import risk_bit
    dq = [qq for qq in q if qq.domain == "DHBT"]
    rate = []
    for qq in dq:
        bits = [risk_bit((recs[i].answers or {}).get(qq.question_id), qq.risk_answer) for i in ite[neg]]
        bits = [b for b in bits if b is not None]
        rate.append(float(np.mean(bits)))
    out["dhbt_item_rate_neg"] = rate
    say(f"R1  chạm {out['trig']:.3f} = chỉ lĩnh vực chính {out['main_only']:.3f} + chỉ dấu hiệu "
        f"{out['dhbt_only']:.3f} + cả hai {out['both']:.3f}; ở trẻ không mắc {out['trig_neg']:.3f}")
    for b in out["by_age"]:
        say(f"    {b['lo']}–{b['hi']} ngày (n {b['n']}): chạm|âm {b['trig_neg']:.3f} · lĩnh vực chính {b['main_neg']:.3f}"
            f" · dấu hiệu {b['dhbt_neg']:.3f}")
    say(f"    tỷ lệ dấu hiệu từng câu DHBT ở trẻ không mắc: {[round(x, 3) for x in rate]}")
    return out


# ------------------------------------------------------------------ R2
def part_poland(say):
    from ts_external import QCHAT_ITEMS, load_poland
    from ts_fusion import domain_risk_counts
    from scipy.stats import spearmanr
    Q, yp, meta = load_poland()
    risk = (Q >= 3).astype(float)                   # thang 0–4 đã mã hoá theo chiều nguy cơ
    pol = {}
    for d in SHARED:
        idx = [i - 1 for i, (dd, _) in QCHAT_ITEMS.items() if dd == d]
        share = risk[:, idx].mean(axis=1)
        pol[d] = {"n_items": len(idx), "neg": float(share[yp == 0].mean()), "pos": float(share[yp == 1].mean()),
                  "auc": auc(yp, share),
                  "rule23_neg": float((share[yp == 0] >= 2 / 3).mean()) if d != "DHBT" else None,
                  "any_neg": float((share[yp == 0] > 0).mean())}
    # mô phỏng: đoàn hệ chính, cả hai nửa, tuổi 18–24 tháng như đoàn hệ Ba Lan
    q = rs._STATE["questions"]
    recs = rs._STATE["cohort"]
    sel = [r for r in recs if 540 <= r.age_days < 750]
    ys = np.array([r.label for r in sel])
    sim = {}
    for d in SHARED:
        sh = []
        for r in sel:
            c = domain_risk_counts(r, q).get(d)
            sh.append(c[0] / c[1] if c and c[1] else np.nan)
        sh = np.array(sh)
        ok = np.isfinite(sh)
        sim[d] = {"n_items": sum(1 for qq in q if qq.domain == d), "neg": float(np.nanmean(sh[ys == 0])),
                  "pos": float(np.nanmean(sh[ys == 1])), "auc": auc(ys[ok], sh[ok]),
                  "rule23_neg": float((sh[(ys == 0) & ok] >= 2 / 3).mean()) if d != "DHBT" else None,
                  "any_neg": float((sh[(ys == 0) & ok] > 0).mean())}
    c1 = all(x[d]["pos"] > x[d]["neg"] for x in (pol, sim) for d in SHARED)
    rho = float(spearmanr([pol[d]["auc"] for d in SHARED], [sim[d]["auc"] for d in SHARED])[0])
    ratios = {d: sim[d]["neg"] / pol[d]["neg"] for d in SHARED}
    c3 = all(0.5 <= v <= 2 for v in ratios.values())
    out = {"poland": pol, "sim": sim, "n_poland": meta["n"], "n_poland_pos": meta["n_pos"],
           "n_sim": len(sel), "n_sim_pos": int(ys.sum()), "C1_same_direction": c1,
           "C2_spearman_domain_auc": rho, "C3_ratio_neg": ratios, "C3_pass": c3}
    say("R2  lĩnh vực | Ba Lan: âm / dương / AUC | mô phỏng 18–24 tháng: âm / dương / AUC")
    for d in SHARED:
        say(f"    {d:<6} {pol[d]['neg']:.3f} / {pol[d]['pos']:.3f} / {pol[d]['auc']:.3f}   |   "
            f"{sim[d]['neg']:.3f} / {sim[d]['pos']:.3f} / {sim[d]['auc']:.3f}   tỷ số âm {ratios[d]:.2f}")
    say(f"    C1 cùng chiều: {c1} · C2 Spearman AUC lĩnh vực {rho:.2f} · C3 cùng bậc: {c3}")
    return out


# ------------------------------------------------------------------ R3
def part_exclusion(say):
    from ts_simulate import generate_dataset
    from ts_features import build_feature_vector
    from ts_fuzzy import build_all_fis, infer_all_domains
    from ts_fusion import byt_threshold_met
    q = rs._STATE["questions"]
    recs, metas = generate_dataset(rs.N_RAW, q, seed=rs.SEED_COHORT)

    def complete(thr):
        rs._apply({**rs.BASELINE, "MISS_THRESHOLD_DOM": thr})
        fis = build_all_fis()
        keep = []
        for r in recs:
            try:
                fv = build_feature_vector(r, q)
            except Exception:
                keep.append(False)
                continue
            risk = infer_all_domains(fv, fis)
            keep.append(all(risk.get(d) is not None for d in cfg.DOMAINS))
        return np.array(keep)

    k02, k05 = complete(0.20), complete(0.50)
    y = np.array([r.label for r in recs])
    age = np.array([r.age_days for r in recs])
    trig = np.array([byt_threshold_met(r, q)[0] for r in recs])
    n_main = int(k02.sum())
    if n_main != len(rs._STATE["cohort"]):
        raise SystemExit("đoàn hệ dựng lại không khớp đoàn hệ chính")

    def desc(m):
        return {"n": int(m.sum()), "prev": float(y[m].mean()), "age_mean": float(age[m].mean()),
                "age_lt_365": float((age[m] < 365).mean()), "trig": float(trig[m].mean())}
    out = {"kept": desc(k02), "excluded": desc(~k02), "kept05": desc(k05)}
    # chấm cấu hình mặc định trên đoàn hệ nới (ngưỡng 0,5)
    main_cohort = rs._STATE["cohort"]
    rs._install_cohort([r for r, k in zip(recs, k05) if k])
    lv = {**rs.BASELINE, "MISS_THRESHOLD_DOM": 0.50}
    raw, net = rs.score_cohort(lv)
    yt = rs._STATE["y"][rs._STATE["ite"]].astype(int)
    tt = rs._STATE["trig"]
    out["loose_cohort"] = {"n_test": int(len(yt)), "n_pos": int(yt.sum()), "trig": float(tt.mean()),
                           "E1": auc(yt, raw), "E2": auc(yt, net), "E3": auc(yt[tt], raw[tt]),
                           "share_at_floor": float(np.isclose(net[tt], cfg.SAFETY_FLOOR).mean())}
    rs._install_cohort(main_cohort)
    say(f"R3  giữ {out['kept']['n']} (mắc {out['kept']['prev']:.4f}, tuổi TB {out['kept']['age_mean']:.0f} ngày, "
        f"chạm {out['kept']['trig']:.3f}) · loại {out['excluded']['n']} (mắc {out['excluded']['prev']:.4f}, "
        f"tuổi TB {out['excluded']['age_mean']:.0f}, <12 tháng {out['excluded']['age_lt_365']:.3f}, "
        f"chạm {out['excluded']['trig']:.3f})")
    lc = out["loose_cohort"]
    say(f"    đoàn hệ nới (0,5): kiểm thử {lc['n_test']} · E1 {lc['E1']:.3f} · E2 {lc['E2']:.3f} · E3 {lc['E3']:.3f}"
        f" · dồn về sàn {lc['share_at_floor']:.4f}")
    return out


# ------------------------------------------------------------------ R4
def part_floor(gen):
    """Gọi trong tiến trình con có SENS_GEN đặt sẵn: chấm cấu hình mặc định, in JSON."""
    rs.init_worker()
    raw, net = rs.score_cohort(dict(rs.BASELINE))
    tt = rs._STATE["trig"]
    y = rs._STATE["y"][rs._STATE["ite"]].astype(int)
    from ts_fusion import domain_risk_counts
    q = rs._STATE["questions"]
    sel = [r for r in rs._STATE["cohort"] if 540 <= r.age_days < 750 and r.label == 0]
    td = {}
    for d in SHARED:
        sh = [c[0] / c[1] for c in (domain_risk_counts(r, q).get(d) for r in sel) if c and c[1]]
        td[d] = float(np.mean(sh))
    print("JSON:" + json.dumps({"gen": gen, "share_at_floor": float(np.isclose(net[tt], cfg.SAFETY_FLOOR).mean()),
                                "td_rate_18_24m": td,
                                "trig": float(tt.mean()), "rule_sens": float(tt[y == 1].mean()),
                                "E1": auc(y, raw), "E2": auc(y, net), "E3": auc(y[tt], raw[tt])}))


def part_scen(say):
    import run_p12_tasks as rt
    prim = json.load(open(os.path.join(REP, "sensitivity_n50k.json"), encoding="utf-8"))
    T = json.load(open(os.path.join(REP, "p12_tasks.json"), encoding="utf-8"))
    out = {}
    for key, gen, label in SCEN:
        d = json.load(open(os.path.join(REP, f"sensitivity_r1_{key}.json"), encoding="utf-8"))
        sob = rt.sobol_run("r1_" + key, d)
        axes = sob["axes"]
        res = {"label": label, "gen": gen, "meta": {k: d["meta"][k] for k in
               ("n_cohort", "n_test", "n_pos_test", "n_trig", "ty_le_duong", "ty_le_cham_byt", "n_config")},
               "auc_default": d["auc_mac_dinh"], "net_loss": d["auc_mac_dinh"]["E1"] - d["auc_mac_dinh"]["E2"]}
        for crit in ("E1", "E2", "E3"):
            st = dict(zip(axes, sob[crit]["ST"]))
            ci = dict(zip(axes, sob[crit]["ST_ci"]))
            real = [a for a in axes if a not in rs.SHAM_AXES]
            floor = max(ci[a][1] for a in rs.SHAM_AXES)
            top = sorted(real, key=lambda a: -st[a])
            tot = sum(st[a] for a in real)
            res[crit] = {"top3": top[:3], "top2_share": (st[top[0]] + st[top[1]]) / tot,
                         "share_agg_dhbt": (st["AGG_OP"] + st["DHBT_LAG_TABLE"]) / tot,
                         "n_indist": sum(ci[a][0] <= floor for a in real), "floor": floor,
                         "oob_r2": sob[crit]["oob_r2"],
                         "bh": sorted(r["truc"] for r in d["phan_tich"][crit]["truc"] if r["y_nghia_BH"]),
                         "sham_bh": any(r["y_nghia_BH"] for r in d["phan_tich"][crit]["truc"] if r["gia_lap"]),
                         "ST": st}
            from scipy.stats import spearmanr
            pst = T["sobol"]["n50k"][crit]["ST"]
            res[crit]["spearman_vs_primary"] = float(spearmanr(pst, [st[a] for a in T["sobol"]["n50k"]["axes"]])[0])
        env = dict(os.environ, SENS_GEN=gen, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--part", "floor", "--gen", gen], env=env,
                           capture_output=True, text=True, encoding="utf-8")
        line = next((x for x in r.stdout.splitlines() if x.startswith("JSON:")), None)
        if line is None:
            raise SystemExit("lỗi kịch bản " + key + ":\n" + r.stderr[-2000:])
        res["floor"] = json.loads(line[5:])
        if abs(res["floor"]["E1"] - d["auc_mac_dinh"]["E1"]) > 1e-12:
            raise SystemExit(f"kịch bản {key}: AUC mặc định không khớp lần quét")
        out[key] = res
        say(f"R4  {label}: chạm {d['meta']['ty_le_cham_byt']:.3f} · mắc {d['meta']['ty_le_duong']:.4f} · "
            f"E1/E2/E3 {d['auc_mac_dinh']['E1']:.3f}/{d['auc_mac_dinh']['E2']:.3f}/{d['auc_mac_dinh']['E3']:.3f} · "
            f"mất do lưới {res['net_loss']:.3f} · dồn về sàn {res['floor']['share_at_floor']:.4f}")
        for crit in ("E1", "E2", "E3"):
            x = res[crit]
            say(f"    {crit}: ba trục đầu {x['top3']} · AGG+DHBT {x['share_agg_dhbt']:.3f} · không khác giả lập "
                f"{x['n_indist']}/18 · BH {len(x['bh'])} · giả lập qua BH {x['sham_bh']} · Spearman với lượt chính "
                f"{x['spearman_vs_primary']:.2f} · R² OOB {x['oob_r2']:.3f}")
    out["primary_trig"] = prim["meta"]["ty_le_cham_byt"]
    return out


def main():
    part = sys.argv[sys.argv.index("--part") + 1]
    if part == "floor":
        part_floor(sys.argv[sys.argv.index("--gen") + 1])
        return
    t0 = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    d = load_out()
    if part == "data":
        rs.no_power_throttling()
        rs.init_worker()
        d["R1"] = part_trigger(say)
        try:
            d["R2"] = part_poland(say)
        except Exception as e:                     # tệp Ba Lan không được phân phối lại trong gói mã
            if "R2" not in d:
                raise
            say(f"R2  bỏ qua ({type(e).__name__}: {e}); giữ kết quả R2 đã có trong {OUT}")
        d["R3"] = part_exclusion(say)
    elif part == "scen":
        d["R4"] = part_scen(say)
    d.setdefault("meta", {})[part] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0, 1)}
    save_out(d)
    open(os.path.join(REP, f"p12_r1_{part}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"xong {part} trong {time.time() - t0:.0f} s → {OUT}")


if __name__ == "__main__":
    main()
