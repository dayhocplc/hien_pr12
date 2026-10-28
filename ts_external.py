# -*- coding: utf-8 -*-
"""
ts_external.py — Kiểm thử ngoài trên đoàn hệ lâm sàng Ba Lan (Q-CHAT-25).

    python src/ts_external.py

VÌ SAO CẦN TỆP NÀY
------------------
V4 (n = 26, nhãn 22/4, không có tập độc lập) là điểm yếu lớn nhất của luận văn.
Đoàn hệ Ba Lan là dữ liệu THẬT duy nhất có thể chạm tới nó: 252 trẻ 18–24 tháng,
nhãn lâm sàng (ADOS-2/ADI-R/DSM-5), lớp gần cân bằng 135/117, dữ liệu từng mục.

Quan trọng: nhãn KHÔNG suy ra được từ bộ câu hỏi — ngưỡng tổng điểm tốt nhất chỉ
đạt accuracy 0,7619. Đối lập hoàn toàn với bộ Toddler Q-CHAT-10 (n = 1054), nơi
nhãn đúng bằng `sum(A1..A10) > 3` ở cả 1054/1054 ca (xem `verify_toddler_leakage`).

BA GIỚI HẠN PHẢI CÔNG BỐ Ở CHƯƠNG 5
-----------------------------------
1. Q-CHAT không có mục vận động nào → chỉ kiểm được 4/6 lĩnh vực. Số hạng tương
   tác dư thừa I(VDTHO, VDTINH) = −0,15 KHÔNG được kiểm chứng ở đây: chưa được
   xác nhận mà cũng chưa bị bác bỏ.
2. Q-CHAT không gắn mốc tuổi kỳ vọng → không tính được LAG_i. Vì vậy FIS lớp 1
   bị bỏ qua (xem cfg.FIS_BYPASS_WHEN_NO_LAG và [REVIEW-P10.3]).
3. Thiết kế bệnh–chứng, không phải đoàn hệ sàng lọc: 135/117 không phải tỷ lệ
   hiện mắc, nên Se/Sp ở đây KHÔNG suy ra được cho sàng lọc cộng đồng.

BẢN QUYỀN
---------
Q-CHAT thuộc Autism Research Centre (Cambridge). Tệp này chỉ mô tả NỘI DUNG HÀNH
VI của từng mục bằng tiếng Việt để phục vụ ánh xạ lĩnh vực, không sao chép nguyên
văn bộ câu hỏi. Khi đưa vào phụ lục luận văn phải ghi rõ nguồn và bản quyền.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

import config as cfg
from ts_aggregate import choquet_2additive, owa, renormalize_for_missing
from ts_evaluate import average_precision, classification_metrics, roc_auc, wilson_ci

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLAND_SAV = os.path.join(PROJECT_DIR, "data", "external", "qchat_poland_252.sav")
TODDLER_CSV = os.path.join(PROJECT_DIR, "data", "external", "toddler_qchat10_2018.csv")


# =====================================================================
# Ánh xạ 25 mục Q-CHAT → 6 lĩnh vực BYT
# =====================================================================
# Nguồn thứ tự mục: phụ lục Allison và cs. (2008) đối chiếu Bảng 2 của bài báo
# Ba Lan đi kèm bộ dữ liệu. Ánh xạ lĩnh vực do nhóm nghiên cứu thực hiện theo
# nội dung hành vi; cần được chuyên gia lâm sàng rà soát trước khi công bố.

QCHAT_ITEMS: dict[int, tuple[str, str]] = {
    1:  ("CNXH",  "Nhìn khi được gọi tên"),
    2:  ("CNXH",  "Giao tiếp bằng mắt — dễ hay khó"),
    3:  ("DHBT",  "Xếp đồ vật thành hàng"),
    4:  ("GTNN",  "Người ngoài hiểu được lời trẻ nói"),
    5:  ("GTNN",  "Chỉ tay để đòi thứ mình muốn"),
    6:  ("GTNN",  "Chỉ tay để chia sẻ sự chú ý"),
    7:  ("DHBT",  "Bị vật quay hút chú ý kéo dài"),
    8:  ("GTNN",  "Số lượng từ nói được"),
    9:  ("BCHOC", "Chơi giả vờ"),
    10: ("BCHOC", "Nhìn theo hướng người khác nhìn"),
    11: ("DHBT",  "Ngửi hoặc liếm vật lạ"),
    12: ("GTNN",  "Dùng tay người lớn như công cụ"),
    13: ("DHBT",  "Đi nhón gót"),
    14: ("DHBT",  "Thích nghi khi đổi nếp sinh hoạt"),
    15: ("CNXH",  "An ủi người khác khi họ buồn"),
    16: ("DHBT",  "Lặp đi lặp lại một hành động"),
    17: ("GTNN",  "Mức điển hình của những từ đầu tiên"),
    18: ("GTNN",  "Nhại lại lời nghe được"),
    19: ("GTNN",  "Dùng cử chỉ đơn giản"),
    20: ("DHBT",  "Cử động ngón tay bất thường gần mắt"),
    21: ("CNXH",  "Nhìn mặt người lớn để dò phản ứng"),
    22: ("DHBT",  "Chỉ tập trung vào một hai đồ vật"),
    23: ("DHBT",  "Xoay/vê đồ vật lặp lại"),
    24: ("DHBT",  "Quá nhạy cảm với tiếng động"),
    25: ("DHBT",  "Nhìn chằm chằm vô định"),
}

# Bộ 10 mục rút gọn chính thức (Allison, Auyeung, Baron-Cohen 2012), ngưỡng ≥ 3/10.
QCHAT10_ITEMS = [1, 2, 5, 6, 9, 10, 15, 17, 19, 25]

# Q-CHAT không có mục nào cho hai lĩnh vực này.
DOMAINS_NOT_COVERED = ["VDTHO", "VDTINH"]

QCHAT_MAX = 4.0        # thang trả lời 0–4 sau khi mã hoá lại theo chiều nguy cơ


class ExternalDataError(RuntimeError):
    pass


# =====================================================================
# Nạp dữ liệu
# =====================================================================

def load_poland(path: str = None, fix_out_of_range: bool = True) -> tuple:
    """→ (Q, y, meta) với Q là ma trận 252×25 thang 0–4, y = 1 nếu nhãn ASD.

    Làm sạch: mục 4 có đúng 1 giá trị 11 nằm ngoài thang 0–4; thay bằng trung vị
    của chính mục đó và ghi lại trong `meta`.
    """
    try:
        import pyreadstat
    except ImportError:
        raise ExternalDataError(
            "Cần gói pyreadstat để đọc tệp SPSS: pip install pyreadstat") from None

    path = path or POLAND_SAV
    if not os.path.exists(path):
        raise ExternalDataError(f"Không tìm thấy {path}")

    df, sav_meta = pyreadstat.read_sav(path)
    cols = [f"qchat{i}recode" for i in range(1, 26)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ExternalDataError(f"Thiếu cột {missing}")

    fixes = {}
    if fix_out_of_range:
        for c in cols:
            bad = df[c] > QCHAT_MAX
            if bad.any():
                med = float(df.loc[~bad, c].median())
                fixes[c] = {"n": int(bad.sum()),
                            "values": df.loc[bad, c].tolist(), "replaced_with": med}
                df.loc[bad, c] = med

    labels = sav_meta.variable_value_labels.get("group", {})
    asd_code = next((k for k, v in labels.items() if str(v).lower().startswith("asd")), 1.0)
    y = (df["group"] == asd_code).astype(int).to_numpy()
    Q = df[cols].to_numpy(dtype=float)

    meta = {
        "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "age_min": float(df["age"].min()), "age_max": float(df["age"].max()),
        "value_labels": labels, "out_of_range_fixes": fixes,
        "n_missing": int(df[cols].isna().sum().sum()),
    }
    return Q, y, meta


def verify_toddler_leakage(path: str = None) -> dict:
    """Kiểm định rò rỉ nhãn của bộ Toddler Q-CHAT-10 (n = 1054).

    Đây là lý do bộ đó KHÔNG được dùng cho bất kỳ con số hiệu năng nào, và là
    minh chứng cụ thể cho cảnh báo "vòng lặp tự xác nhận" ở §8.1.
    """
    import csv
    path = path or TODDLER_CSV
    if not os.path.exists(path):
        raise ExternalDataError(f"Không tìm thấy {path}")

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    col = next(c for c in rows[0] if c.strip().startswith("Class"))
    n_bad_sum = n_bad_rule = 0
    for r in rows:
        total = int(r["Qchat-10-Score"])
        items = sum(int(r[f"A{i}"]) for i in range(1, 11))
        is_yes = r[col].strip() == "Yes"
        if total != items:
            n_bad_sum += 1
        if (total > 3) != is_yes:
            n_bad_rule += 1

    return {"n": len(rows),
            "n_positive": sum(1 for r in rows if r[col].strip() == "Yes"),
            "violations_score_equals_sum": n_bad_sum,
            "violations_label_is_threshold": n_bad_rule,
            "deterministic": n_bad_sum == 0 and n_bad_rule == 0}


# =====================================================================
# Đặc trưng lĩnh vực từ Q-CHAT
# =====================================================================

def domain_scores(Q: np.ndarray, items: Sequence[int] = None
                  ) -> dict[str, np.ndarray]:
    """d_i ∈ [0,1] cho từng lĩnh vực, trung bình các mục thuộc lĩnh vực đó / 4.

    Thang 0–4 nên d_i LIÊN TỤC thật sự — nghịch lý là bộ Ba Lan kiểm chứng tính
    "mờ thật" (V1) tốt hơn chính dữ liệu Việt Nam mức lĩnh vực.
    """
    items = list(range(1, 26)) if items is None else list(items)
    out: dict[str, np.ndarray] = {}
    for domain in cfg.DOMAINS:
        idx = [i - 1 for i in items if QCHAT_ITEMS[i][0] == domain]
        if idx:
            out[domain] = Q[:, idx].mean(axis=1) / QCHAT_MAX
    return out


def to_rows(d: dict[str, np.ndarray]) -> list[dict[str, float]]:
    n = len(next(iter(d.values())))
    return [{k: float(v[i]) for k, v in d.items()} for i in range(n)]


# =====================================================================
# Các cách tổng hợp đem ra so sánh
# =====================================================================

def aggregate(rows: list[dict[str, float]], method: str,
              shapley: dict = None, interaction: dict = None) -> np.ndarray:
    domains = list(rows[0].keys())
    shapley = cfg.SHAPLEY if shapley is None else shapley
    interaction = cfg.INTERACTION if interaction is None else interaction
    phi, itr = renormalize_for_missing(shapley, interaction, domains)

    if method == "uniform":
        return np.array([float(np.mean(list(x.values()))) for x in rows])
    if method == "shapley":
        return np.array([sum(phi[k] * v for k, v in x.items()) for x in rows])
    if method == "choquet":
        return np.array([choquet_2additive(x, phi, itr) for x in rows])
    if method == "owa":
        return np.array([owa(x, cfg.OWA_WEIGHTS) for x in rows])
    raise ValueError(f"Cách tổng hợp không hỗ trợ: {method!r}")


# =====================================================================
# Thống kê suy luận
# =====================================================================

def bootstrap_auc_ci(y, s, n_boot: int = 3000, seed: int = 20260827):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        a = roc_auc(y[i].tolist(), s[i].tolist())
        if a is not None:
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_auc_test(y, s_a, s_b, n_boot: int = 3000, seed: int = 20260827):
    """Bootstrap phân tầng cho hiệu AUC(a) − AUC(b) → (hiệu, CI95, p hai phía)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        a = roc_auc(y[i].tolist(), s_a[i].tolist())
        b = roc_auc(y[i].tolist(), s_b[i].tolist())
        if a is not None and b is not None:
            diffs.append(a - b)
    diffs = np.array(diffs)
    obs = roc_auc(y.tolist(), s_a.tolist()) - roc_auc(y.tolist(), s_b.tolist())
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return obs, (float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))), p


def item_auc_table(Q: np.ndarray, y: np.ndarray) -> list[dict]:
    """AUC đơn biến của từng mục — dùng cho §2.2 (gán w_j từ y văn có trích dẫn)."""
    out = []
    for i in range(1, 26):
        domain, text = QCHAT_ITEMS[i]
        a = roc_auc(y.tolist(), Q[:, i - 1].tolist())
        out.append({"item": i, "domain": domain, "text": text, "auc": a,
                    "in_qchat10": i in QCHAT10_ITEMS,
                    "reversed": a < 0.5})
    return out


# =====================================================================
# Báo cáo
# =====================================================================

def run_report() -> int:
    def hr(t):
        print()
        print("=" * 96)
        print(t)
        print("=" * 96)

    hr("BỘ TODDLER Q-CHAT-10 — KIỂM ĐỊNH RÒ RỈ NHÃN")
    try:
        leak = verify_toddler_leakage()
        print(f"  n = {leak['n']}, nhãn dương = {leak['n_positive']} "
              f"({leak['n_positive']/leak['n']:.1%})")
        print(f"  Số ca vi phạm 'điểm = tổng 10 mục'      : "
              f"{leak['violations_score_equals_sum']}/{leak['n']}")
        print(f"  Số ca vi phạm 'nhãn = (điểm > 3)'        : "
              f"{leak['violations_label_is_threshold']}/{leak['n']}")
        if leak["deterministic"]:
            print("\n  → Nhãn là HÀM TẤT ĐỊNH của chính các mục dùng làm đặc trưng.")
            print("    Mọi mô hình chỉ cần học lại phép cộng và một ngưỡng.")
            print("    KHÔNG dùng bộ này cho bất kỳ con số hiệu năng nào (§8.1).")
    except ExternalDataError as exc:
        print(f"  (bỏ qua: {exc})")

    hr("ĐOÀN HỆ BA LAN — NẠP VÀ KIỂM TRA DỮ LIỆU")
    Q, y, meta = load_poland()
    print(f"  n = {meta['n']}   ASD = {meta['n_pos']}   chứng = {meta['n_neg']}  "
          f"({meta['n_pos']/meta['n']:.1%} dương tính)")
    print(f"  tuổi {meta['age_min']:.0f}–{meta['age_max']:.0f} tháng   "
          f"ô thiếu: {meta['n_missing']}")
    print(f"  nhãn: {meta['value_labels']}")
    for c, f in meta["out_of_range_fixes"].items():
        print(f"  ⚠ {c}: {f['n']} giá trị ngoài thang 0–4 ({f['values']}) "
              f"→ thay bằng trung vị {f['replaced_with']}")

    total = Q.sum(axis=1)
    best_t, best_acc = max(
        ((t, float(((total >= t).astype(int) == y).mean())) for t in range(0, 101)),
        key=lambda kv: kv[1])
    print(f"\n  Ngưỡng tổng điểm tốt nhất: ≥ {best_t} → accuracy {best_acc:.4f}")
    print(f"  → Nhãn KHÔNG suy ra được từ bộ câu hỏi (khác hẳn bộ Toddler). "
          "Dùng được làm tập kiểm thử ngoài.")

    hr("AUC ĐƠN BIẾN TỪNG MỤC (phục vụ §2.2 — gán w_j từ y văn)")
    table = item_auc_table(Q, y)
    print(f"  {'#':>3s} {'nội dung':38s} {'BYT':7s} {'AUC':>7s}  {'Q10':>4s}")
    print("  " + "-" * 66)
    for r in table:
        flag = " ⚠ ngược" if r["reversed"] else ""
        print(f"  {r['item']:3d} {r['text']:38s} {r['domain']:7s} {r['auc']:7.4f}  "
              f"{'✔' if r['in_qchat10'] else '':>4s}{flag}")

    hr("NĂNG LỰC THEO NHÓM LĨNH VỰC")
    for name, doms in [("Giao tiếp – xã hội (GTNN+CNXH+BCHOC)",
                        ["GTNN", "CNXH", "BCHOC"]),
                       ("Hành vi lặp lại (DHBT)", ["DHBT"])]:
        idx = [i - 1 for i in range(1, 26) if QCHAT_ITEMS[i][0] in doms]
        s = Q[:, idx].sum(axis=1)
        a = roc_auc(y.tolist(), s.tolist())
        lo, hi = bootstrap_auc_ci(y, s)
        print(f"  {name:40s} {len(idx):2d} mục  AUC = {a:.4f} [{lo:.4f}; {hi:.4f}]")
    print("\n  → Căn cứ định lượng cho việc hạ φ_DHBT ở [REVIEW-P10.1].")

    hr("KIỂM THỬ NGOÀI: TÁCH BẠCH 'CHỌN CÂU HỎI' VỚI 'CÁCH TỔNG HỢP'")
    print("  Áp CÙNG bộ tổng hợp lên CÙNG bộ mục, để hai yếu tố không lẫn vào nhau.\n")

    results = {}
    for tag, items in [("25 mục (đầy đủ)", list(range(1, 26))),
                       ("10 mục (Q-CHAT-10)", QCHAT10_ITEMS)]:
        rows = to_rows(domain_scores(Q, items))
        doms = list(rows[0].keys())
        base = Q[:, [i - 1 for i in items]].sum(axis=1)
        variants = {
            "1. Cộng thô từng mục": base,
            "2. Gộp lĩnh vực, trọng số đều": aggregate(rows, "uniform"),
            "3. Gộp lĩnh vực, Shapley": aggregate(rows, "shapley"),
            "4. + Choquet (tương tác lâm sàng)":
                aggregate(rows, "choquet", interaction=cfg.INTERACTION_CLINICAL),
            "5. OWA": aggregate(rows, "owa"),
        }
        results[tag] = variants
        print(f"  --- {tag} · {len(doms)} lĩnh vực: {doms} ---")
        for lab, s in variants.items():
            a = roc_auc(y.tolist(), list(s))
            lo, hi = bootstrap_auc_ci(y, s)
            extra = ""
            if lab != "1. Cộng thô từng mục":
                o, ci, p = paired_auc_test(y, s, base)
                extra = f"   vs dòng 1: {o:+.4f} [{ci[0]:+.4f}; {ci[1]:+.4f}] p={p:.3f}"
            print(f"    {lab:36s} AUC = {a:.4f} [{lo:.4f}; {hi:.4f}]{extra}")
        print()

    hr("HAI YẾU TỐ ĐÓNG GÓP BAO NHIÊU, VÀ CÓ CỘNG DỒN KHÔNG")
    a25 = roc_auc(y.tolist(), list(results["25 mục (đầy đủ)"]["1. Cộng thô từng mục"]))
    a10 = roc_auc(y.tolist(), list(results["10 mục (Q-CHAT-10)"]["1. Cộng thô từng mục"]))
    s25 = roc_auc(y.tolist(), list(results["25 mục (đầy đủ)"]["3. Gộp lĩnh vực, Shapley"]))
    s10 = roc_auc(y.tolist(), list(results["10 mục (Q-CHAT-10)"]["3. Gộp lĩnh vực, Shapley"]))
    print(f"  Chỉ chọn câu hỏi  : {a25:.4f} → {a10:.4f}   = {a10 - a25:+.4f}")
    print(f"  Chỉ đổi tổng hợp  : {a25:.4f} → {s25:.4f}   = {s25 - a25:+.4f}")
    print(f"  Làm cả hai        : {a25:.4f} → {s10:.4f}   = {s10 - a25:+.4f}")
    print("\n  → Hai yếu tố KHÔNG cộng dồn. Gộp lĩnh vực là cơ chế KHỬ NHIỄU:")
    print("    nó có giá trị đúng khi phiếu chứa câu yếu, và hết tác dụng khi")
    print("    bộ câu hỏi đã được lọc sạch. Phiếu 20 câu BYT thuộc trường hợp đầu")
    print("    (6/20 câu vào trục vận động mà §1.2 ghi là ít đặc hiệu).")

    hr("ĐỘ NHẠY THEO TRỌNG SỐ φ_DHBT")
    rows = to_rows(domain_scores(Q))
    base_phi = {"GTNN": 0.263, "CNXH": 0.263, "BCHOC": 0.197}
    for w in (0.195, 0.150, 0.100, 0.080, 0.050, 0.020, 0.0):
        rest = 1.0 - w
        sc = rest / sum(base_phi.values())
        phi = {k: v * sc for k, v in base_phi.items()}
        phi["DHBT"] = w
        s = np.array([sum(phi[k] * v for k, v in x.items()) for x in rows])
        mark = "  ← giá trị đã chọn" if abs(w - cfg.SHAPLEY["DHBT"]) < 1e-9 else ""
        print(f"  φ_DHBT = {w:.3f}  →  AUC = {roc_auc(y.tolist(), s.tolist()):.4f}{mark}")
    print("\n  Tăng đơn điệu khi φ → 0, nhưng KHÔNG đặt về 0: xem [REVIEW-P10.1].")

    hr("GIỚI HẠN CỦA KIỂM THỬ NGOÀI NÀY")
    print(f"  • Q-CHAT không có mục nào cho {DOMAINS_NOT_COVERED} → chỉ kiểm được 4/6")
    print("    lĩnh vực. Số hạng I(VDTHO, VDTINH) = −0,15 CHƯA được kiểm chứng.")
    print("  • Không có mốc tuổi kỳ vọng → không tính được LAG_i; FIS lớp 1 bị bỏ qua.")
    print("  • Thiết kế bệnh–chứng 135/117, không phải tỷ lệ hiện mắc → Se/Sp ở đây")
    print("    KHÔNG suy ra được cho sàng lọc cộng đồng.")
    print()
    print("  " + cfg.DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_report())
