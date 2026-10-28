# -*- coding: utf-8 -*-
"""
ts_evaluate.py — §9 ĐÁNH GIÁ MÔ HÌNH.

Với n = 26, điểm ước lượng đơn lẻ không có giá trị: MỌI tỷ lệ báo cáo phải kèm
khoảng tin cậy Wilson (§0.3.6, §9.3). Trình bày chủ động điều đó được đánh giá
là nghiêm túc về phương pháp; để hội đồng phát hiện thì thành điểm trừ nặng.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import config as cfg


# =====================================================================
# §9.3 — khoảng tin cậy Wilson. BẮT BUỘC.
# =====================================================================

def wilson_ci(k: int, n: int, z: float = cfg.WILSON_Z) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỷ lệ k/n.

    Với Sp = 3/4 (chỉ 4 ca âm tính), khoảng rộng tới [30,1 % ; 95,4 %] — chính
    độ rộng đó là nội dung cần bàn luận, không phải thứ cần giấu.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(k: int, n: int, z: float = cfg.WILSON_Z) -> str:
    """'96,15 % (25/26) CI 95 % Wilson = [81,1 % ; 99,3 %]'"""
    if n <= 0:
        return "—"
    lo, hi = wilson_ci(k, n, z)
    return (f"{100 * k / n:.2f} % ({k}/{n}) "
            f"CI 95 % = [{100 * lo:.1f} % ; {100 * hi:.1f} %]")


# =====================================================================
# §9.2 — nhóm chỉ số phân loại
# =====================================================================

def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Se, Sp, PPV, NPV, F1, Accuracy, Balanced Accuracy — mỗi tỷ lệ kèm CI Wilson.

    Sp và NPV đang THIẾU trong luận văn hiện tại (§9.2); với một công cụ sàng
    lọc, NPV là con số phụ huynh thực sự cần biết.
    """
    cm = confusion(y_true, y_pred)
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]

    def rate(k: int, n: int) -> Optional[dict]:
        if n <= 0:
            return None
        lo, hi = wilson_ci(k, n)
        return {"value": k / n, "k": k, "n": n, "ci": (lo, hi)}

    se = rate(tp, tp + fn)
    sp = rate(tn, tn + fp)
    ppv = rate(tp, tp + fp)
    npv = rate(tn, tn + fn)
    acc = rate(tp + tn, len(y_true))

    f1 = None
    if ppv and se and (ppv["value"] + se["value"]) > 0:
        f1 = 2 * ppv["value"] * se["value"] / (ppv["value"] + se["value"])
    bal_acc = None
    if se and sp:
        bal_acc = (se["value"] + sp["value"]) / 2

    return {"confusion": cm, "Se": se, "Sp": sp, "PPV": ppv, "NPV": npv,
            "Accuracy": acc, "F1": f1, "BalancedAccuracy": bal_acc,
            "n": len(y_true)}


def format_metrics(metrics: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))
    cm = metrics["confusion"]
    lines.append(f"  TP={cm['TP']}  FN={cm['FN']}  FP={cm['FP']}  TN={cm['TN']}"
                 f"   (n = {metrics['n']})")
    for key, label in [("Se", "Se  (độ nhạy)"), ("Sp", "Sp  (độ đặc hiệu)"),
                       ("PPV", "PPV"), ("NPV", "NPV"), ("Accuracy", "Accuracy")]:
        m = metrics[key]
        if m is None:
            lines.append(f"  {label:20s} —  (không tính được)")
        else:
            lines.append(f"  {label:20s} {fmt_ci(m['k'], m['n'])}")
    if metrics["F1"] is not None:
        lines.append(f"  {'F1':20s} {metrics['F1']:.4f}")
    if metrics["BalancedAccuracy"] is not None:
        lines.append(f"  {'Balanced Accuracy':20s} {metrics['BalancedAccuracy']:.4f}")
    return "\n".join(lines)


# =====================================================================
# §9.5 — phân tầng theo độ chắc chắn. MỤC QUAN TRỌNG NHẤT.
# =====================================================================

def stratified_by_confidence(confidences: Sequence[float],
                             y_true: Sequence[int],
                             y_pred: Sequence[int],
                             threshold: float = cfg.CONF_THRESHOLD) -> dict:
    """Chia hồ sơ thành tầng CHẮC CHẮN / RANH GIỚI rồi đo riêng từng tầng.

    Luận điểm cần chứng minh bằng bảng này: sai sót của mô hình tập trung ở
    nhóm mà chính hệ thống đã tự nhận diện là chưa chắc chắn — nói cách khác,
    mô hình biết khi nào nó không biết. Phương pháp cộng tổng điểm không có khả
    năng này.
    """
    strata: dict[str, dict[str, list]] = {
        "chac_chan": {"y_true": [], "y_pred": []},
        "ranh_gioi": {"y_true": [], "y_pred": []},
    }
    for conf, t, p in zip(confidences, y_true, y_pred):
        key = "chac_chan" if conf >= threshold else "ranh_gioi"
        strata[key]["y_true"].append(t)
        strata[key]["y_pred"].append(p)

    n_total = len(y_true)
    out = {"threshold": threshold, "n_total": n_total, "strata": {}}
    for name, data in strata.items():
        n = len(data["y_true"])
        out["strata"][name] = {
            "n": n,
            "share": n / n_total if n_total else 0.0,
            "metrics": classification_metrics(data["y_true"], data["y_pred"]) if n else None,
        }
    return out


def stratified_screening(model_scores: Sequence[float],
                         safety_triggered: Sequence[bool],
                         y_true: Sequence[int],
                         threshold: float = cfg.CONF_THRESHOLD,
                         floor: float = cfg.SAFETY_FLOOR) -> dict:
    """§9.5 — BẢN SỬA (T1 / L17). Phân tầng theo độ chắc chắn CỦA MÔ HÌNH.

    Lỗi của bản cũ: độ chắc chắn được tính trên điểm SAU lưới an toàn. Lưới đặt
    mọi hồ sơ chạm ngưỡng BYT về đúng SAFETY_FLOOR = 65, nơi
    μ_NGHI_NGO − μ_CAN_THEO_DOI = 0,41 − 0,17 = 0,245 < CONF_THRESHOLD, nên mọi
    hồ sơ bị lưới nâng đều rơi vào tầng RANH GIỚI và đều bị gọi dương tính.
    Accuracy của tầng đó vì thế đúng bằng tỷ lệ hiện mắc trong tầng (≈ 4 %),
    không nói gì về việc mô hình "biết khi nào nó không biết".

    Bản sửa tách ba tầng:
      • "ghi_de"    — lưới an toàn nâng điểm (chạm BYT và điểm mô hình < sàn):
                      kết luận do QUY TẮC đưa ra, không thuộc về độ chắc chắn
                      của mô hình, nên báo cáo riêng;
      • "chac_chan" / "ranh_gioi" — các hồ sơ còn lại, độ chắc chắn và kết luận
                      (mức NGHI_NGO) tính trên CHÍNH điểm của mô hình.
    """
    from ts_fusion import confidence_of, fuzzify_output, level_of

    strata = {k: {"y_true": [], "y_pred": []}
              for k in ("ghi_de", "chac_chan", "ranh_gioi")}
    for s, trig, t in zip(model_scores, safety_triggered, y_true):
        if trig and s < floor:
            key, pred = "ghi_de", 1
        else:
            conf = confidence_of(fuzzify_output(s))
            key = "chac_chan" if conf >= threshold else "ranh_gioi"
            pred = 1 if level_of(s) == "NGHI_NGO" else 0
        strata[key]["y_true"].append(int(t))
        strata[key]["y_pred"].append(pred)

    n_total = len(y_true)
    out = {"threshold": threshold, "n_total": n_total, "strata": {}}
    for name, data in strata.items():
        n = len(data["y_true"])
        errors = sum(int(a != b) for a, b in zip(data["y_true"], data["y_pred"]))
        out["strata"][name] = {
            "n": n,
            "share": n / n_total if n_total else 0.0,
            "n_pos": sum(data["y_true"]),
            "errors": errors,
            "metrics": classification_metrics(data["y_true"], data["y_pred"]) if n else None,
        }
    return out


# =====================================================================
# §9.4 — đường hiệu chỉnh
# =====================================================================

def calibration_curve(y_true: Sequence[int], scores: Sequence[float],
                      n_bins: int = cfg.CALIB_BINS) -> list[dict]:
    """Chia điểm thành n_bins khoảng, trả tỷ lệ dương tính thực tế theo điểm
    dự đoán trung bình. Đường lý tưởng là đường chéo."""
    lo, hi = 0.0, 100.0
    width = (hi - lo) / n_bins
    bins: list[dict] = []
    for i in range(n_bins):
        left = lo + i * width
        right = left + width
        idx = [j for j, s in enumerate(scores)
               if (left <= s < right) or (i == n_bins - 1 and s == right)]
        if not idx:
            bins.append({"bin": (left, right), "n": 0,
                         "mean_score": None, "observed_rate": None})
            continue
        bins.append({
            "bin": (left, right),
            "n": len(idx),
            "mean_score": sum(scores[j] for j in idx) / len(idx),
            "observed_rate": sum(y_true[j] for j in idx) / len(idx),
        })
    return bins


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """(1/n) Σ (p_i − y_i)² — chất lượng hiệu chỉnh."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    return sum((p - t) ** 2 for t, p in zip(y_true, probabilities)) / n


def expected_calibration_error(y_true: Sequence[int], probabilities: Sequence[float],
                               n_bins: int = cfg.CALIB_BINS) -> float:
    """ECE — sai số hiệu chỉnh kỳ vọng."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n_bins):
        left, right = i / n_bins, (i + 1) / n_bins
        idx = [j for j, p in enumerate(probabilities)
               if (left <= p < right) or (i == n_bins - 1 and p == right)]
        if not idx:
            continue
        conf = sum(probabilities[j] for j in idx) / len(idx)
        acc = sum(y_true[j] for j in idx) / len(idx)
        total += len(idx) / n * abs(conf - acc)
    return total


# =====================================================================
# Chỉ số liên tục — chỉ có được nhờ đầu ra mờ (§9.2)
# =====================================================================

def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """AUC bằng thống kê Mann–Whitney U, xử lý đúng trường hợp điểm bằng nhau."""
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """PR-AUC — phù hợp hơn ROC khi mất cân bằng lớp.

    Các quan sát có cùng điểm số được gộp thành MỘT bậc thang. Nếu cộng dồn
    tp/rank theo từng quan sát, thứ tự ngẫu nhiên bên trong nhóm trùng điểm sẽ
    tự nó sinh ra một phần diện tích — sai lệch tới 0,029 so với sklearn khi
    điểm bị làm tròn. Nhánh Fuzzy k-NN với k = 3 chỉ phát ra vài giá trị thuộc
    (0, 1/3, 2/3, 1) nên gần như mọi quan sát đều trùng điểm.
    """
    n_pos = sum(y_true)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n = len(order)
    total = 0.0
    tp = 0
    rec_prev = 0.0
    for pos, i in enumerate(order):
        tp += y_true[i]
        # Chỉ chốt một bậc tại quan sát CUỐI CÙNG của mỗi nhóm trùng điểm.
        if pos + 1 < n and scores[order[pos + 1]] == scores[i]:
            continue
        prec = tp / (pos + 1)
        rec = tp / n_pos
        total += prec * (rec - rec_prev)
        rec_prev = rec
    return total
