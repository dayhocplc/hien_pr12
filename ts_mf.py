# -*- coding: utf-8 -*-
"""
ts_mf.py — Hàm thuộc và các phép toán mờ cơ bản.

Tự cài đặt thay vì gọi scikit-fuzzy vì hai lý do:
  1. §10.2 — công thức tự cài được thì trình bày được trong luận văn.
  2. [REVIEW-P7.3] — hàm thuộc dạng VAI (shoulder) như trimf[0, 0, 0.8] hoặc
     trapmf[1.5, 2.5, 3, 3] trả về 0 tại chính điểm mút nếu cài theo công thức
     tam giác/hình thang thông thường (x ≤ a → 0). Lỗi này làm TOÀN BỘ đầu ra
     FIS bằng 0 mà không báo lỗi gì. Ở đây xử lý tường minh và có test MF-5.

Tham chiếu: §3.1, §3.2, §7.4.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class MembershipError(ValueError):
    """Tham số hàm thuộc không hợp lệ (MF-2, MF-3)."""


# =====================================================================
# Hàm thuộc cơ bản
# =====================================================================

def trimf(x: float, params: Sequence[float]) -> float:
    """Hàm thuộc tam giác μ(x; a, b, c).

    Quy ước tại điểm mút giống scikit-fuzzy: μ(b) = 1 kể cả khi a == b
    (vai trái) hoặc b == c (vai phải).
    """
    if len(params) != 3:
        raise MembershipError(f"trimf cần đúng 3 tham số, nhận {len(params)}")   # MF-3
    a, b, c = (float(v) for v in params)
    if not (a <= b <= c):
        raise MembershipError(f"trimf yêu cầu a <= b <= c, nhận {params}")
    if a == c:
        raise MembershipError(f"trimf suy biến (a == c): {params}")              # MF-2

    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0


def trapmf(x: float, params: Sequence[float]) -> float:
    """Hàm thuộc hình thang μ(x; a, b, c, d), quy ước vai giống trimf."""
    if len(params) != 4:
        raise MembershipError(f"trapmf cần đúng 4 tham số, nhận {len(params)}")  # MF-3
    a, b, c, d = (float(v) for v in params)
    if not (a <= b <= c <= d):
        raise MembershipError(f"trapmf yêu cầu a <= b <= c <= d, nhận {params}")
    if a == d:
        raise MembershipError(f"trapmf suy biến (a == d): {params}")

    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


_KIND_TO_FUNC = {"trimf": trimf, "trapmf": trapmf}


def membership(spec: tuple, x: float) -> float:
    """Tính độ thuộc từ khai báo dạng ("trimf", [a, b, c]) trong config."""
    kind, params = spec
    try:
        func = _KIND_TO_FUNC[kind]
    except KeyError:
        raise MembershipError(f"Kiểu hàm thuộc không hỗ trợ: {kind!r}") from None
    return func(float(x), params)


def membership_array(spec: tuple, xs: np.ndarray) -> np.ndarray:
    """Bản vector hoá, dùng khi giải mờ trên universe rời rạc."""
    return np.array([membership(spec, float(v)) for v in xs], dtype=float)


def support(spec: tuple) -> tuple[float, float]:
    """Giá đỡ (support) của tập mờ — dùng để kiểm tra chồng lấn MF-1."""
    _, params = spec
    return float(params[0]), float(params[-1])


# =====================================================================
# t-norm / s-norm (§3.1)
# =====================================================================

def t_norm(a: float, b: float, kind: str = "product") -> float:
    """Phép AND mờ.

    Mặc định `product`, KHÔNG phải `min`: với `min`, trọng số kích hoạt bão hoà
    ở 1.0 và đầu ra bị kẹt ở một số ít giá trị — một trong hai nguyên nhân của
    vấn đề V1 nêu ở §0.2.
    """
    if kind == "product":
        return a * b
    if kind == "min":
        return min(a, b)
    raise MembershipError(f"t-norm không hỗ trợ: {kind!r}")


def s_norm(a: float, b: float, kind: str = "prob_sum") -> float:
    """Phép OR mờ.

    `bounded_sum` (t-conorm Łukasiewicz) là lựa chọn duy nhất trong ba lựa chọn
    này bảo toàn được tính đơn điệu của FIS lớp 1 khi hàm thuộc đầu vào tạo
    thành phân hoạch đơn vị — xem [REVIEW-P12] và `test_fuzzy_mf.py`.
    """
    if kind == "prob_sum":
        return a + b - a * b
    if kind == "bounded_sum":
        return min(1.0, a + b)
    if kind == "max":
        return max(a, b)
    raise MembershipError(f"s-norm không hỗ trợ: {kind!r}")


def partition_deficit(mf: dict, lo: float, hi: float, n: int = 2001) -> float:
    """Độ lệch lớn nhất của Σ_k μ_k(x) so với 1 trên toàn universe (MF-6).

    §3.2 chỉ yêu cầu MF-4 là Σ μ_k(x) > 0 — điều kiện quá yếu. Khi tổng độ
    thuộc tụt xuống dưới 1 trong vùng chồng lấn, tổng trọng số kích hoạt cũng
    tụt theo, và đầu ra giải mờ có thể GIẢM khi đầu vào TĂNG. Xem [REVIEW-P12].
    """
    import numpy as _np
    xs = _np.linspace(lo, hi, n)
    sums = [sum(membership(spec, float(x)) for spec in mf.values()) for x in xs]
    return float(max(abs(s - 1.0) for s in sums))


# =====================================================================
# Giải mờ (§3.1)
# =====================================================================

def centroid(universe: np.ndarray, mu: np.ndarray) -> float | None:
    """Giải mờ trọng tâm. Trả None khi tập mờ kết quả rỗng (không luật nào kích hoạt)."""
    total = float(np.sum(mu))
    if total <= 1e-12:
        return None
    return float(np.sum(universe * mu) / total)


def make_universe(lo: float, hi: float, step: float) -> np.ndarray:
    """Universe rời rạc [lo, hi] có chứa cả hai điểm mút.

    Điểm mút phải nằm trong universe: với trapmf[1.5, 2.5, 3, 3] thì μ = 1 chỉ
    tại x = 3, nên bỏ sót điểm cuối sẽ mất toàn bộ vùng nhân của tập.
    """
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


# =====================================================================
# Kiểm tra bắt buộc MF-1..MF-5 (§3.2)
# =====================================================================

def check_overlap(spec_a: tuple, spec_b: tuple) -> float:
    """Độ dài phần giao của hai giá đỡ. MF-1 yêu cầu > 0 với mọi cặp liền kề."""
    a_lo, a_hi = support(spec_a)
    b_lo, b_hi = support(spec_b)
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def check_coverage(mf_set: dict, lo: float, hi: float, step: float) -> list[float]:
    """MF-4 — trả về danh sách các điểm mà Σ_k μ_k(x) = 0 (lý tưởng: rỗng)."""
    gaps = []
    for x in make_universe(lo, hi, step):
        total = sum(membership(spec, float(x)) for spec in mf_set.values())
        if total <= 1e-12:
            gaps.append(float(x))
    return gaps


def validate_mf_set(mf_set: dict, lo: float, hi: float, step: float,
                    name: str = "") -> dict:
    """Chạy MF-1..MF-5 trên một bộ hàm thuộc, trả về báo cáo.

    MF-1  mọi cặp liền kề (theo thứ tự khai báo) chồng lấn thật
    MF-2  không có hàm suy biến
    MF-3  đúng số tham số
    MF-4  phủ kín universe
    MF-5  [REVIEW-P7.3] hàm vai đạt μ = 1 tại điểm mút của universe
    """
    keys = list(mf_set.keys())
    report = {"name": name, "ok": True, "errors": [], "warnings": []}

    # MF-2, MF-3 — hàm thuộc tự kiểm khi được gọi lần đầu.
    for key in keys:
        try:
            membership(mf_set[key], lo)
        except MembershipError as exc:
            report["ok"] = False
            report["errors"].append(f"MF-2/MF-3 [{key}]: {exc}")

    # MF-1
    for left, right in zip(keys, keys[1:]):
        width = check_overlap(mf_set[left], mf_set[right])
        if width <= 0:
            report["ok"] = False
            report["errors"].append(
                f"MF-1 [{left} | {right}]: giao rỗng, không phải hệ mờ thật")
        else:
            report.setdefault("overlaps", {})[f"{left}|{right}"] = width

    # MF-4
    gaps = check_coverage(mf_set, lo, hi, step)
    if gaps:
        report["ok"] = False
        report["errors"].append(
            f"MF-4: {len(gaps)} điểm không được phủ, ví dụ {gaps[:5]}")

    # MF-5 — tại hai điểm mút phải có ít nhất một tập đạt μ = 1.
    for edge in (lo, hi):
        best = max(membership(spec, edge) for spec in mf_set.values())
        if best < 1.0 - 1e-9:
            report["ok"] = False
            report["errors"].append(
                f"MF-5: tại x = {edge} độ thuộc lớn nhất chỉ đạt {best:.4f}; "
                "hàm vai nhiều khả năng bị cài sai")

    return report
