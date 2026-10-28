# -*- coding: utf-8 -*-
"""
ts_aggregate.py — §4 NHÁNH 1: TỔNG HỢP BẰNG TÍCH PHÂN CHOQUET 2-CỘNG TÍNH.

Khắc phục V3: đây là thành phần cho phép nói "mô hình phân tích tổ hợp hành vi"
một cách có cơ sở toán học. Không dùng FIS ở lớp 2 vì 6 đầu vào × 3 hàm thuộc =
729 tổ hợp luật, không thể viết tay (§4.1).

[REVIEW-P2] — CẢNH BÁO VỀ CÔNG THỨC
------------------------------------
Đặc tả §4.2 đưa ra hai dạng và nói dạng thứ hai "trực quan hơn để cài đặt".
HAI DẠNG ĐÓ KHÔNG TƯƠNG ĐƯƠNG. Dạng thứ hai trong đặc tả viết:

    C(x) = Σ φ'_i·x_i + Σ_{I>0} I_ij·min − Σ_{I<0} |I_ij|·max
    φ'_i = φ_i − ½ Σ_{j≠i} I_ij                       ← thiếu trị tuyệt đối

Khai triển đúng (dùng |x_i−x_j| = max − min và min + max = x_i + x_j) là:

    C(x) = Σ φ'_i·x_i + Σ_{I>0} I_ij·min + Σ_{I<0} |I_ij|·max
    φ'_i = φ_i − ½ Σ_{j≠i} |I_ij|                     ← TRỊ TUYỆT ĐỐI

Tại x = (GTNN 0, CNXH 0, BCHOC 0, DHBT 0, VDTINH 0, VDTHO 1) với bộ tham số
Phụ lục A: dạng chuẩn cho 0,1550 còn dạng sai cho 0,0050 — lệch 31 lần, đúng
trên cặp có tương tác âm VDTHO/VDTINH, tức đúng chức năng mà Choquet được đưa
vào để làm.

Nguy hiểm hơn cả: CH-1..CH-4 KHÔNG bắt được lỗi này (dạng sai vẫn cho C(0)=0,
C(1)=1 và 0 vi phạm đơn điệu trên 2×10⁵ cặp Monte Carlo). Vì vậy dưới đây cài
dạng chuẩn làm bản chính thức, giữ dạng khai triển để trình bày trong luận văn,
và thêm CH-6 đối chiếu hai dạng.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

import config as cfg
from ts_types import ValidationReport


# =====================================================================
# Truy cập chỉ số tương tác
# =====================================================================

def interaction_of(interaction: Mapping[tuple[str, str], float],
                   a: str, b: str) -> float:
    """I_ij đối xứng; cặp không khai báo mặc định bằng 0."""
    if (a, b) in interaction:
        return interaction[(a, b)]
    if (b, a) in interaction:
        return interaction[(b, a)]
    return 0.0


# =====================================================================
# §4.2 — Tích phân Choquet 2-cộng tính
# =====================================================================

def choquet_2additive(x: Mapping[str, float],
                      shapley: Mapping[str, float],
                      interaction: Mapping[tuple[str, str], float]) -> float:
    """DẠNG CHUẨN — bản cài đặt chính thức.

        C(x) = Σ_i φ_i·x_i − ½ Σ_{i<j} I_ij·|x_i − x_j|

    x: {'GTNN': 0.78, ...} ∈ [0,1] → C(x) ∈ [0,1].
    Ngắn và không có chỗ để sai dấu; xem cảnh báo P2 ở đầu tệp.
    """
    domains = list(x.keys())
    missing = [d for d in domains if d not in shapley]
    if missing:
        raise KeyError(f"Thiếu giá trị Shapley cho: {missing}")

    total = sum(shapley[d] * x[d] for d in domains)
    for a, b in itertools.combinations(domains, 2):
        total -= 0.5 * interaction_of(interaction, a, b) * abs(x[a] - x[b])
    return total


def choquet_2additive_expanded(x: Mapping[str, float],
                               shapley: Mapping[str, float],
                               interaction: Mapping[tuple[str, str], float]) -> float:
    """DẠNG KHAI TRIỂN — chỉ dùng để trình bày và để CH-6 đối chiếu.

        C(x) = Σ_i φ'_i·x_i
             + Σ_{I_ij > 0} I_ij·min(x_i, x_j)        ← cộng hưởng
             + Σ_{I_ij < 0} |I_ij|·max(x_i, x_j)      ← dư thừa
        φ'_i = φ_i − ½ Σ_{j≠i} |I_ij|

    Đây là dạng ĐÚNG, khác với dạng in trong đặc tả v1.0 (xem P2).
    """
    domains = list(x.keys())
    total = 0.0
    for d in domains:
        corr = sum(abs(interaction_of(interaction, d, other))
                   for other in domains if other != d)
        total += (shapley[d] - 0.5 * corr) * x[d]

    for a, b in itertools.combinations(domains, 2):
        i_ab = interaction_of(interaction, a, b)
        if i_ab > 0:
            total += i_ab * min(x[a], x[b])
        elif i_ab < 0:
            total += abs(i_ab) * max(x[a], x[b])
    return total


def choquet_spec_v1_buggy(x: Mapping[str, float],
                          shapley: Mapping[str, float],
                          interaction: Mapping[tuple[str, str], float]) -> float:
    """Dạng in trong đặc tả v1.0 — GIỮ LẠI CHỈ ĐỂ CHỨNG MINH LÀ SAI.

    Không gọi hàm này ở bất kỳ đâu ngoài test. Nó tồn tại để luận văn có thể
    dẫn con số cụ thể khi trình bày việc phát hiện và sửa lỗi.
    """
    domains = list(x.keys())
    total = 0.0
    for d in domains:
        corr = sum(interaction_of(interaction, d, other)
                   for other in domains if other != d)
        total += (shapley[d] - 0.5 * corr) * x[d]

    for a, b in itertools.combinations(domains, 2):
        i_ab = interaction_of(interaction, a, b)
        if i_ab > 0:
            total += i_ab * min(x[a], x[b])
        elif i_ab < 0:
            total -= abs(i_ab) * max(x[a], x[b])
    return total


# =====================================================================
# §4.5 — Phương án dự phòng OWA
# =====================================================================

def owa(x: Mapping[str, float], weights: Sequence[float]) -> float:
    """Ordered Weighted Averaging: sắp giảm dần rồi lấy tổ hợp có trọng số.

    v = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04] diễn đạt mềm mệnh đề "chỉ cần một
    vài lĩnh vực bất thường rõ là đủ đáng lo" — tinh thần của quy định BYT
    nhưng không có ngưỡng cứng.
    """
    values = sorted(x.values(), reverse=True)
    w = list(weights[:len(values)])
    if not w:
        raise ValueError("OWA cần ít nhất một trọng số")
    w_sum = sum(w)
    if w_sum <= 0:
        raise ValueError("Tổng trọng số OWA phải dương")
    w = [v / w_sum for v in w]      # chuẩn hoá lại khi có lĩnh vực bị loại
    return float(sum(wi * vi for wi, vi in zip(w, values)))


# =====================================================================
# §4.4 — Ràng buộc CH-1..CH-6
# =====================================================================

def validate_measure(shapley: Mapping[str, float],
                     interaction: Mapping[tuple[str, str], float],
                     n_monte_carlo: int = cfg.MEASURE_MC_SAMPLES,
                     seed: int = 12345) -> ValidationReport:
    """Kiểm CH-1..CH-6. Gọi lúc khởi động ứng dụng.

    Một độ đo mờ vi phạm đơn điệu nghĩa là TĂNG nguy cơ ở một lĩnh vực có thể
    làm GIẢM điểm tổng — lỗi toán học, không phải lựa chọn thiết kế.
    """
    report = ValidationReport(name="Kiểm tra độ đo mờ CH-1..CH-6", ok=True)
    domains = list(shapley.keys())
    tol = cfg.MEASURE_TOL

    # CH-1 — Σ φ_i = 1
    total_phi = sum(shapley.values())
    report.checks["CH-1_sum_phi"] = total_phi
    if abs(total_phi - 1.0) > tol:
        report.ok = False
        report.errors.append(f"CH-1: Σφ = {total_phi:.12f} ≠ 1.0")

    # CH-2 — φ_i ≥ ½ Σ_{j≠i} |I_ij|  (điều kiện đủ cho đơn điệu)
    ch2 = {}
    for d in domains:
        bound = 0.5 * sum(abs(interaction_of(interaction, d, o))
                          for o in domains if o != d)
        ch2[d] = {"phi": shapley[d], "bound": bound, "margin": shapley[d] - bound}
        if shapley[d] + tol < bound:
            report.ok = False
            report.errors.append(
                f"CH-2 [{d}]: φ = {shapley[d]:.4f} < ½Σ|I| = {bound:.4f} "
                "→ độ đo không đơn điệu")
    report.checks["CH-2"] = ch2
    tightest = min(ch2.items(), key=lambda kv: kv[1]["margin"])
    report.checks["CH-2_tightest"] = {"domain": tightest[0], **tightest[1]}

    # CH-3 — điều kiện biên
    zeros = {d: 0.0 for d in domains}
    ones = {d: 1.0 for d in domains}
    c0 = choquet_2additive(zeros, shapley, interaction)
    c1 = choquet_2additive(ones, shapley, interaction)
    report.checks["CH-3"] = {"C(0)": c0, "C(1)": c1}
    if abs(c0) > tol or abs(c1 - 1.0) > tol:
        report.ok = False
        report.errors.append(f"CH-3: C(0) = {c0:.12f}, C(1) = {c1:.12f}")

    # CH-4 — đơn điệu Monte Carlo
    rng = np.random.default_rng(seed)
    n_violations = 0
    worst = 0.0
    for _ in range(n_monte_carlo):
        lo = rng.random(len(domains))
        hi = lo + rng.random(len(domains)) * (1.0 - lo)
        x = dict(zip(domains, lo))
        y = dict(zip(domains, hi))
        gap = choquet_2additive(y, shapley, interaction) - \
            choquet_2additive(x, shapley, interaction)
        if gap < -1e-12:
            n_violations += 1
            worst = min(worst, gap)
    report.checks["CH-4"] = {"n_samples": n_monte_carlo,
                             "n_violations": n_violations,
                             "worst_gap": worst}
    if n_violations:
        report.ok = False
        report.errors.append(
            f"CH-4: {n_violations}/{n_monte_carlo} cặp vi phạm đơn điệu "
            f"(lệch xấu nhất {worst:.2e})")

    # CH-6 [REVIEW-P2] — dạng chuẩn và dạng khai triển phải trùng nhau.
    max_gap = 0.0
    for _ in range(n_monte_carlo):
        x = dict(zip(domains, rng.random(len(domains))))
        gap = abs(choquet_2additive(x, shapley, interaction)
                  - choquet_2additive_expanded(x, shapley, interaction))
        max_gap = max(max_gap, gap)
    report.checks["CH-6_max_gap"] = max_gap
    if max_gap > 1e-9:
        report.ok = False
        report.errors.append(
            f"CH-6: hai dạng Choquet lệch tới {max_gap:.3e} — một trong hai sai dấu")

    return report


# =====================================================================
# CH-5 — chuẩn hoá lại khi có lĩnh vực bị loại
# =====================================================================

def renormalize_for_missing(shapley: Mapping[str, float],
                            interaction: Mapping[tuple[str, str], float],
                            available: Iterable[str]
                            ) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """CH-5 — giữ nguyên I_ij của các cặp còn hiện diện, chuẩn hoá lại φ.

    Sau khi chuẩn hoá, CH-2 có thể bị phá vỡ (φ nhỏ đi tương đối so với ΣI giữ
    nguyên). Trong trường hợp đó, thu nhỏ toàn bộ I_ij theo cùng một hệ số k —
    đúng thủ pháp đã dùng ở §4.3 — để giữ thứ tự tương đối giữa các cặp.
    """
    available = [d for d in shapley if d in set(available)]
    if not available:
        raise ValueError("Không còn lĩnh vực nào để tổng hợp")

    total = sum(shapley[d] for d in available)
    if total <= 0:
        raise ValueError("Tổng Shapley của các lĩnh vực còn lại không dương")
    phi = {d: shapley[d] / total for d in available}

    keep = {}
    for (a, b), v in interaction.items():
        if a in phi and b in phi:
            keep[(a, b)] = v

    # Hệ số thu nhỏ lớn nhất vẫn thoả CH-2 trên tập còn lại.
    k = 1.0
    for d in phi:
        bound = 0.5 * sum(abs(interaction_of(keep, d, o)) for o in phi if o != d)
        if bound > 0:
            k = min(k, phi[d] / bound)
    if k < 1.0:
        keep = {pair: v * k for pair, v in keep.items()}

    return phi, keep


def aggregate_rule_branch(x: Mapping[str, float],
                          shapley: Mapping[str, float] = None,
                          interaction: Mapping[tuple[str, str], float] = None,
                          operator: str = "choquet") -> float:
    """s₁ = 100 · C(x). `operator` ∈ {"choquet", "owa"} — dùng cho ablation A6′."""
    shapley = cfg.SHAPLEY if shapley is None else shapley
    interaction = cfg.INTERACTION if interaction is None else interaction

    phi, itr = renormalize_for_missing(shapley, interaction, x.keys())
    if operator == "owa":
        return 100.0 * owa(x, cfg.OWA_WEIGHTS)
    if operator == "choquet":
        return 100.0 * choquet_2additive(x, phi, itr)
    raise ValueError(f"Toán tử tổng hợp không hỗ trợ: {operator!r}")
