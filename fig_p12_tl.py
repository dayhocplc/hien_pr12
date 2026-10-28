# -*- coding: utf-8 -*-
r"""
fig_p12_tl.py — bốn hình của bản thảo Bài 12, góc nhìn công nghệ hỗ trợ tâm lý trị liệu
(manuscript/P12_TL; với --lang en: manuscript/P12_TL_EN).

Hình 1  Đường tính điểm của công cụ, các tham số gắn vào từng tầng, ba điểm đo E1/E2/E3.
Hình 2  Chỉ số Sobol toàn phần (KTC 95 %) ở E1 và E3, tô màu theo nhóm trách nhiệm; ngưỡng giả lập.
Hình 3  Đảo hạng tầm quan trọng trước (E1) và sau (E2) lưới an toàn.
Hình 4  Hàng đợi khám trong nhóm đã chạm ngưỡng: (a) phân phối điểm hệ thống báo ra; (b) tỷ lệ trẻ có
        rối loạn được đánh giá theo phần hàng đợi đã khám.

Chạy:  .venv/Scripts/python.exe src/fig_p12_tl.py [--lang en]
       (cần out/reports/sensitivity_n50k.json, p12_tasks.json, p12_tl.json)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EN = "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en"
R1 = "--r1" in sys.argv          # bản sửa theo phản biện MS.22: thuật ngữ "hồ sơ dương tính", lưu vào manuscript/P12_R1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "out", "reports")
OUT = os.path.join(ROOT, "manuscript", "P12_R1" if R1 else ("P12_TL_EN" if EN else "P12_TL"))
PREFIX = "p12r1_fig" if R1 else ("p12tl_en_fig" if EN else "p12tl_fig")
S = json.load(open(os.path.join(REP, "sensitivity_n50k.json"), encoding="utf-8"))
T = json.load(open(os.path.join(REP, "p12_tasks.json"), encoding="utf-8"))
TL = json.load(open(os.path.join(REP, "p12_tl.json"), encoding="utf-8"))


def tr(vi, en):
    return en if EN else vi


INK, MUTED = "#222222", "#666666"
BLUE, ORANGE, GREEN, VERM, PURPLE, GREY, LIGHT = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                  "#666666", "#BBBBBB")
AXIS_VI = {
    "AGG_OP": "Toán tử tổng hợp lĩnh vực", "DHBT_LAG_TABLE": "Bảng luật hành vi bất thường",
    "DEFUZZ": "Phép giải mờ", "MF_SET": "Hàm thuộc đầu vào",
    "SHAPLEY": "Trọng số lĩnh vực (Shapley)", "INTERACTION": "Chỉ số tương tác",
    "LAG_MAX": "Trần chỉ số trễ tuổi", "MISS_THRESHOLD_DOM": "Ngưỡng câu bỏ trống",
    "DHBT_ZERO_AS_MISSING": "Coi 'không dấu hiệu' là thiếu",
    "AGG_POINT_ESTIMATE": "Ước lượng điểm của khoảng", "SAFETY_FLOOR": "Sàn lưới an toàn",
    "FUSION_WEIGHTS": "Trọng số hợp nhất", "FKNN_K": "k-NN mờ: số láng giềng", "FKNN_M": "k-NN mờ: số mũ",
    "FCM_C": "C-Means mờ: số cụm", "FCM_M": "C-Means mờ: số mũ", "T_NORM": "Phép AND mờ (t-norm)",
    "S_NORM": "Phép OR mờ (s-norm)", "GIA_LAP_A": "Trục giả lập A", "GIA_LAP_B": "Trục giả lập B",
}
AXIS_EN = {
    "AGG_OP": "Domain aggregation operator", "DHBT_LAG_TABLE": "Rule table, abnormal behaviour",
    "DEFUZZ": "Defuzzification", "MF_SET": "Input membership functions",
    "SHAPLEY": "Domain weights (Shapley)", "INTERACTION": "Interaction indices",
    "LAG_MAX": "Age-lag cap", "MISS_THRESHOLD_DOM": "Missing-answer threshold",
    "DHBT_ZERO_AS_MISSING": "'No signs' as missing",
    "AGG_POINT_ESTIMATE": "Interval point estimate", "SAFETY_FLOOR": "Safety-net floor",
    "FUSION_WEIGHTS": "Fusion weights", "FKNN_K": "Fuzzy k-NN: k", "FKNN_M": "Fuzzy k-NN: fuzzifier",
    "FCM_C": "Fuzzy C-means: c", "FCM_M": "Fuzzy C-means: fuzzifier", "T_NORM": "Fuzzy AND (t-norm)",
    "S_NORM": "Fuzzy OR (s-norm)", "GIA_LAP_A": "Sham axis A", "GIA_LAP_B": "Sham axis B",
}
AXIS = AXIS_EN if EN else AXIS_VI
SHAM = ("GIA_LAP_A", "GIA_LAP_B")
GROUP_COL = {"tri_thuc_lam_sang": GREEN, "toan_tu_mo": BLUE, "hoc_tu_du_lieu": PURPLE,
             "quy_dinh_va_hop_le": VERM}
GROUP_LAB = {"tri_thuc_lam_sang": tr("biểu diễn tri thức lâm sàng", "clinical-knowledge representation"),
             "toan_tu_mo": tr("toán tử mờ", "fuzzy operators"),
             "hoc_tu_du_lieu": tr("thành phần học từ dữ liệu", "learned components"),
             "quy_dinh_va_hop_le": tr("quy định và hợp lệ dữ liệu", "regulation and data validity")}
AX_GROUP = {a: g for g, m in TL["tl3"]["groups"].items() for a in m}


def num(x, nd=2):
    s = f"{x:.{nd}f}"
    return s if EN else s.replace(".", ",")


def save(fig, i):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{PREFIX}{i}.png"), dpi=300)
    plt.close(fig)
    print("đã ghi", f"{PREFIX}{i}")


def sob(crit):
    r = T["sobol"]["n50k"][crit]
    return T["sobol"]["n50k"]["axes"], np.array(r["ST"]), np.array(r["ST_ci"])


def fig1():
    fs.setup()
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(-0.01, 1.07)
    ax.set_ylim(-0.05, 1.0)
    ax.axis("off")
    stages = [
        (tr("Phiếu hỏi cha mẹ\n20 câu Có/Không,\n6 lĩnh vực", "Parent-report\nquestionnaire,\n20 yes/no items,\n6 domains"),
         tr(["Ngưỡng câu bỏ trống", "'Không dấu hiệu'", "Ước lượng điểm", "Trần trễ tuổi"],
            ["Missing-answer threshold", "'No signs' as missing", "Point estimate", "Age-lag cap"])),
        (tr("Suy diễn mờ\ntheo lĩnh vực\n(thiếu hụt, trễ)", "Per-domain\nfuzzy inference\n(deficit, age lag)"),
         tr(["AND / OR mờ", "Phép giải mờ", "Hàm thuộc", "Luật hành vi\nbất thường"],
            ["Fuzzy AND / OR", "Defuzzification", "Membership fns", "Abnormal-behaviour\nrules"])),
        (tr("Tổng hợp\n6 lĩnh vực\n(Choquet / OWA)", "Aggregation\nof 6 domains\n(Choquet / OWA)"),
         tr(["Toán tử tổng hợp", "Trọng số lĩnh vực", "Tương tác"],
            ["Aggregation operator", "Domain weights", "Interactions"])),
        (tr("Hợp nhất với\nk-NN mờ và\nC-Means mờ", "Fusion with\nfuzzy k-NN and\nfuzzy C-means"),
         tr(["Trọng số hợp nhất", "k-NN: k, m", "C-Means: c, m"], ["Fusion weights", "k-NN: k, m", "C-means: c, m"])),
        (tr("Lưới an toàn:\nđiểm ≥ sàn\nnếu chạm ngưỡng\nBộ Y tế", "Safety net:\nscore ≥ floor\nif national\nreferral rule met"),
         [tr("Sàn lưới an toàn", "Safety-net floor")]),
    ]
    w, h, y0, gap = 0.165, 0.36, 0.50, 0.047
    xs = [i * (w + gap) for i in range(len(stages))]
    for i, ((title, axes_), x) in enumerate(zip(stages, xs)):
        fs.box(ax, x, y0, w, h, title, fc="#F4F8FB" if i < 4 else "#FCEFE6", ec=BLUE if i < 4 else VERM, fs=7.0)
        ax.text(x + w / 2, y0 - 0.05, "\n".join(axes_), ha="center", va="top", fontsize=6.4, color=MUTED,
                linespacing=1.3)
        if i < len(stages) - 1:
            fs.arrow(ax, (x + w + 0.004, y0 + h / 2), (xs[i + 1] - 0.004, y0 + h / 2))
    e1, e2 = xs[3] + w + gap / 2, xs[4] + w + 0.03
    for x, lab, col, txt in [(e1, "E1", BLUE, tr("trước lưới", "before the net")),
                             (e2, "E2", VERM, tr("điểm hiển thị", "displayed score"))]:
        ax.plot([x, x], [y0 - 0.01, y0 + h + 0.01], color=col, lw=2.0, zorder=4)
        ax.text(x, y0 + h + 0.04, lab, ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
        ax.text(x, y0 + h + 0.13, txt, ha="center", va="bottom", fontsize=6.4, color=col)
    ax.text(1.06, -0.04, tr("E3 = E1 trong nhóm trẻ đã chạm ngưỡng Bộ Y tế\n(mọi trẻ đều được chuyển; điểm chỉ còn\n"
                            "dùng để xếp thứ tự đánh giá)",
                            "E3 = E1 among children who meet the referral rule\n(all are referred; the score only\n"
                            "orders them for assessment)"),
            ha="right", va="bottom", fontsize=6.4, color=GREEN)
    ax.text(0.0, -0.04, tr("Chữ xám: 18 tham số được quét, đặt ở tầng mà chúng tác động.\n"
                           "Hai trục giả lập được lấy mẫu và phân tích như thật nhưng không nối vào đâu.",
                           "Grey: the 18 swept parameters at the stage they act on.\n"
                           "Two sham axes are sampled and analysed like real ones but wired to nothing."),
            ha="left", va="bottom", fontsize=6.4, color=MUTED)
    save(fig, 1)


def fig2():
    fs.setup()
    names, st1, ci1 = sob("E1")
    _, st3, ci3 = sob("E3")
    order = np.argsort(st1)
    y = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 5.0), sharey=True)
    for ax, st, ci, lab, crit in ((axs[0], st1, ci1, tr("(a) Trước lưới an toàn (E1)", "(a) Before the safety net (E1)"), "E1"),
                                  (axs[1], st3, ci3, tr("(b) Trong nhóm đã chạm ngưỡng (E3)",
                                                        "(b) Among referred children (E3)"), "E3")):
        st_o, ci_o = st[order], ci[order]
        cols = [ORANGE if names[i] in SHAM else GROUP_COL[AX_GROUP[names[i]]] for i in order]
        ax.barh(y, np.maximum(st_o, 0), color=cols, height=0.7, edgecolor="none")
        ax.errorbar(np.maximum(st_o, 0), y, xerr=[np.maximum(st_o - ci_o[:, 0], 0), np.maximum(ci_o[:, 1] - st_o, 0)],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
        ax.axvline(TL["tl3"][f"{crit}|floor"], color=ORANGE, ls="--", lw=0.9)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.01, 0.1, 1])
        ax.set_xticklabels(["0", num(0.01), num(0.1, 1), "1"])
        ax.set_xlabel(tr("Chỉ số Sobol toàn phần $S_T$", "Total-order Sobol index $S_T$"))
        ax.set_title(lab, fontsize=8.5, loc="left")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([AXIS[names[i]] for i in order], fontsize=7)
    fig.legend(handles=[Patch(color=GROUP_COL[g], label=GROUP_LAB[g]) for g in GROUP_COL]
               + [Patch(color=ORANGE, label=tr("trục giả lập (nét đứt: ngưỡng nhiễu)", "sham axis (dashed: noise floor)"))],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 2)


def fig3():
    fs.setup()
    names, st1, _ = sob("E1")
    _, st2, _ = sob("E2")
    r1 = len(names) - np.argsort(np.argsort(st1))
    r2 = len(names) - np.argsort(np.argsort(st2))
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    hi = {"SAFETY_FLOOR", "FUSION_WEIGHTS", "AGG_OP", "DHBT_LAG_TABLE", "DEFUZZ", "SHAPLEY", "MF_SET"}
    for i, a in enumerate(names):
        col = ORANGE if a in SHAM else (GROUP_COL[AX_GROUP[a]] if a in hi else LIGHT)
        ax.plot([0, 1], [r1[i], r2[i]], color=col, lw=1.8 if a in hi else 0.9, marker="o", ms=3.5,
                zorder=3 if a in hi else 2)
        tc = col if a in hi or a in SHAM else MUTED
        ax.text(-0.04, r1[i], f"{AXIS[a]}  {num(st1[i], 3)}", ha="right", va="center", fontsize=6.4, color=tc)
        ax.text(1.04, r2[i], f"{num(st2[i], 3)}  {AXIS[a]}", ha="left", va="center", fontsize=6.4, color=tc)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(len(names) + 0.8, 0.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([tr("Hạng theo $S_T$ ở E1\n(trước lưới an toàn)", "Rank by $S_T$, E1\n(before the safety net)"),
                        tr("Hạng theo $S_T$ ở E2\n(điểm hiển thị)", "Rank by $S_T$, E2\n(displayed score)")],
                       fontsize=7.5)
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    save(fig, 3)


def fig4():
    fs.setup()
    h = TL["tl2"]["hist"]
    rows = TL["tl2"]["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axs[0]
    e = np.array(h["edges"])
    c = 0.5 * (e[1:] + e[:-1])
    wdt = e[1] - e[0]
    ax.bar(c, h["not_trig"], width=wdt * 0.9, color=LIGHT, label=tr("không chạm ngưỡng", "rule not met"))
    ax.bar(c, h["trig"], width=wdt * 0.9, color=VERM, bottom=h["not_trig"], label=tr("đã chạm ngưỡng", "rule met (referred)"))
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel(tr("Điểm hệ thống hiển thị (0–100)", "Displayed score (0–100)"))
    ax.set_ylabel("Số hồ sơ (tập kiểm thử)" if R1 else tr("Số trẻ (tập kiểm thử)", "Children (test set)"))
    ax.set_title(tr("(a) Điểm hiển thị, cấu hình mặc định", "(a) Displayed score, default configuration"),
                 fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="upper left")
    ax = axs[1]
    x = np.linspace(0, 1, 51)
    for key, lab, col, ls in (("mac_dinh|tron", tr("điểm hợp nhất, mặc định", "fused score, default"), BLUE, "-"),
                              ("SHAPLEY_v1|tron", tr("điểm hợp nhất, trọng số v1", "fused score, original weights"),
                               GREEN, "--"),
                              ("DEFUZZ_mom|tron", tr("điểm hợp nhất, giải mờ MoM", "fused score, mean of maxima"),
                               PURPLE, "-."),
                              ("mac_dinh|bao_ra", tr("điểm hiển thị (sau lưới)", "displayed score (after the net)"),
                               VERM, "-")):
        ax.plot(100 * x, 100 * np.array(rows[key]["curve"]), color=col, ls=ls, lw=1.4, label=lab)
    ax.axvline(25, color=MUTED, ls=":", lw=0.8)
    if R1:
        ax.set_xlabel("Phần thứ tự đã đi qua (%)")
        ax.set_ylabel("Hồ sơ dương tính đã được xếp lượt (%)")
        ax.set_title("(b) Thứ tự trong nhóm chạm quy tắc chuyển", fontsize=8.5, loc="left")
    else:
        ax.set_xlabel(tr("Phần hàng đợi đã được đánh giá (%)", "Share of the queue assessed (%)"))
        ax.set_ylabel(tr("Trẻ có rối loạn đã được đánh giá (%)", "Affected children assessed (%)"))
        ax.set_title(tr("(b) Hàng đợi trong nhóm đã chạm ngưỡng", "(b) Assessment queue among referred children"),
                     fontsize=8.5, loc="left")
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save(fig, 4)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
