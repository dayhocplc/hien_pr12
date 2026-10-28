# -*- coding: utf-8 -*-
"""
ts_fuzzy.py — §3 FIS LỚP 1: SÁU HỆ MỜ CON.

Khắc phục V2: thay 7 luật phủ thưa trên không gian 5 biến bằng 6 hệ mờ nhỏ,
mỗi hệ phủ kín 100% không gian đầu vào của nó (lưới đầy đủ 3×3 = 9 luật).

Kiểu Mamdani, t-norm `product`, hợp thành `max`, giải mờ trọng tâm.
Đổi `min` → `product` là một trong hai sửa chữa gốc của V1: với `min`, trọng số
kích hoạt bão hoà ở 1.0 và đầu ra bị kẹt ở một số ít giá trị.

Mở rộng [REVIEW-P3]: `infer_feature()` chạy FIS trên TOÀN BỘ tập cấu hình khả dĩ
của một lĩnh vực (xem ts_features) và trả về khoảng [risk_lo, risk_hi] chính xác.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import config as cfg
from ts_mf import (centroid, make_universe, membership, membership_array,
                   s_norm, t_norm, validate_mf_set)
from ts_types import DomainFeature, FeatureVector, Interval, RuleActivation, ValidationReport


def _point_estimate(values: Sequence[float]) -> float:
    if cfg.AGG_POINT_ESTIMATE == "mid":
        return (min(values) + max(values)) / 2.0
    return sum(values) / len(values)


def _interval_from(values: Sequence[float]) -> Interval:
    lo, hi = min(values), max(values)
    point = max(lo, min(hi, _point_estimate(values)))
    return Interval(lo, point, hi)


def _risk_mf_set() -> dict:
    return cfg.MF_RISK_DOMAIN


# =====================================================================
# Hệ mờ con cho một lĩnh vực
# =====================================================================

class DomainFIS:
    """Hệ mờ Mamdani (d_i, LAG_i) → risk_i ∈ [0, 100] cho một lĩnh vực."""

    def __init__(self, domain_code: str,
                 mf_d: dict = None, mf_lag: dict = None,
                 s_norm_kind: str = None):
        """`mf_d`, `mf_lag`, `s_norm_kind` chỉ dùng cho nghiên cứu loại trừ §9.7
        và cho test đối chứng với bộ tham số v1.0 — xem [REVIEW-P12]."""
        if domain_code not in cfg.DOMAINS:
            raise ValueError(f"Lĩnh vực không hợp lệ: {domain_code!r}")
        self.domain = domain_code
        self.mf_d = cfg.MF_D if mf_d is None else mf_d
        self.mf_lag = cfg.MF_LAG if mf_lag is None else mf_lag
        self.s_norm_kind = cfg.S_NORM if s_norm_kind is None else s_norm_kind
        self.mf_risk = _risk_mf_set()

        # §3.3 — lĩnh vực không gắn mốc tuổi dùng bảng luật rút gọn theo d.
        self.ageless = domain_code in cfg.AGELESS_DOMAINS and not cfg.DHBT_USE_LAG_TABLE
        if self.ageless:
            self.rules = [(d_lbl, None, out) for d_lbl, out in cfg.RULES_AGELESS]
        else:
            self.rules = [(a, b, c) for a, b, c in cfg.RULES_DOMAIN]

        self._universe = make_universe(0.0, 100.0, cfg.RES_RISK)
        self._consequent = {
            label: membership_array(spec, self._universe)
            for label, spec in self.mf_risk.items()
        }
        self._cache: dict[tuple[float, float], float] = {}

    # -- suy diễn -----------------------------------------------------

    def _firing(self, d: float, lag: float) -> list[tuple[str, float, str]]:
        """Trọng số kích hoạt của từng luật."""
        out = []
        for d_label, lag_label, risk_label in self.rules:
            w = membership(self.mf_d[d_label], d)
            if lag_label is not None:
                w = t_norm(w, membership(self.mf_lag[lag_label], lag), cfg.T_NORM)
            out.append((d_label, w, risk_label) if lag_label is None
                       else (f"{d_label} ∧ {lag_label}", w, risk_label))
        return out

    def infer(self, d: Optional[float], lag: Optional[float] = 0.0) -> Optional[float]:
        """→ risk_i ∈ [0,100]. Trả None nếu d là None (lĩnh vực bị loại)."""
        if d is None:
            return None
        lag = 0.0 if (lag is None or self.ageless) else float(lag)
        key = (round(float(d), 10), round(lag, 10))
        if key in self._cache:
            return self._cache[key]

        # [REVIEW-P12] Gộp trọng số của các luật CÙNG KẾT LUẬN bằng s-norm
        # TRƯỚC khi cắt, thay vì hợp thành bằng max sau khi cắt.
        #
        # Với `max`, khi d đi qua vùng chồng lấn giữa hai tập đầu vào cùng dẫn
        # tới một kết luận, max(w₁, w₂) tụt xuống dưới cả hai giá trị ở hai đầu
        # — đầu ra GIẢM trong khi đầu vào TĂNG. Đo được: tổng mức giảm 78,2 điểm
        # theo d và 100,7 điểm theo LAG trên lưới quét.
        #
        # `bounded_sum` cộng lại đúng phần bị mất: khi hàm thuộc đầu vào tạo
        # thành phân hoạch đơn vị, tổng trọng số theo mọi kết luận bằng đúng 1,
        # và vi phạm đơn điệu về 0,000. Giá trị biên không đổi.
        by_consequent: dict[str, float] = {}
        for _, w, risk_label in self._firing(float(d), lag):
            if w <= 0.0:
                continue
            prev = by_consequent.get(risk_label)
            by_consequent[risk_label] = (w if prev is None
                                         else s_norm(prev, w, self.s_norm_kind))

        aggregated = np.zeros_like(self._universe)
        for risk_label, w in by_consequent.items():
            # Cắt (Mamdani implication = min) rồi hợp thành bằng max.
            aggregated = np.maximum(aggregated,
                                    np.minimum(w, self._consequent[risk_label]))

        value = centroid(self._universe, aggregated)
        if value is None:
            # MF-4 bảo đảm không xảy ra; nếu xảy ra là lỗi cấu hình hàm thuộc.
            raise RuntimeError(
                f"[{self.domain}] không luật nào kích hoạt tại d={d}, LAG={lag}; "
                "kiểm tra MF-4 (phủ kín universe)")
        self._cache[key] = value
        return value

    def infer_feature(self, feature: DomainFeature) -> Optional[Interval]:
        """Chạy FIS trên mọi cấu hình khả dĩ → khoảng risk chính xác.

        Với dữ liệu từng câu, `combos` có đúng một phần tử nên khoảng suy biến
        thành điểm — trùng khớp hoàn toàn với đặc tả gốc.
        """
        if not feature.available or not feature.combos:
            return None
        values = [self.infer(d, lag) for d, lag in feature.combos]
        return _interval_from([v for v in values if v is not None])

    # -- giải thích ---------------------------------------------------

    def explain(self, d: Optional[float], lag: Optional[float] = 0.0
                ) -> list[RuleActivation]:
        """→ danh sách luật kích hoạt kèm trọng số, sắp giảm dần.

        Bắt buộc theo §3.4: đây là nội dung cho phần "vì sao hệ thống kết luận
        như vậy" trong báo cáo PDF và là bằng chứng trực quan khi bảo vệ.
        """
        if d is None:
            return []
        lag = 0.0 if (lag is None or self.ageless) else float(lag)
        acts = [
            RuleActivation(domain=self.domain,
                           rule_name=f"R{i + 1}",
                           antecedent=antecedent,
                           firing_strength=w,
                           consequent=risk_label)
            for i, (antecedent, w, risk_label) in enumerate(self._firing(float(d), lag))
            if w > 1e-9
        ]
        acts.sort(key=lambda a: a.firing_strength, reverse=True)
        return acts


# =====================================================================
# API mức hệ thống
# =====================================================================

def build_all_fis() -> dict[str, DomainFIS]:
    return {domain: DomainFIS(domain) for domain in cfg.DOMAINS}


def infer_all_domains(features: FeatureVector,
                      fis: Optional[dict[str, DomainFIS]] = None
                      ) -> dict[str, Optional[Interval]]:
    """→ {'GTNN': Interval(78.4), 'VDTHO': Interval(12.1), ...}"""
    fis = fis or build_all_fis()
    return {domain: fis[domain].infer_feature(feat)
            for domain, feat in features.domains.items()}


def risk_to_unit(risk: dict[str, Optional[Interval]], which: str = "point"
                 ) -> dict[str, float]:
    """Chuyển risk ∈ [0,100] về x ∈ [0,1] cho tầng tổng hợp §4.

    `which` ∈ {"lo", "point", "hi"} — dùng "lo"/"hi" để lan truyền khoảng.
    """
    out = {}
    for domain, iv in risk.items():
        if iv is None:
            continue
        out[domain] = getattr(iv, which) / 100.0
    return out


# =====================================================================
# Kiểm tra khởi động — MF-1..MF-5 (§3.2)
# =====================================================================

def validate_membership_functions() -> ValidationReport:
    report = ValidationReport(name="Kiểm tra hàm thuộc MF-1..MF-5", ok=True)

    for name, mf_set, lo, hi, step in [
        ("MF_D", cfg.MF_D, 0.0, 1.0, cfg.RES_D),
        ("MF_LAG", cfg.MF_LAG, 0.0, cfg.LAG_MAX, cfg.RES_LAG),
        ("MF_RISK_DOMAIN", _risk_mf_set(), 0.0, 100.0, cfg.RES_RISK),
        ("MF_OUTPUT", cfg.MF_OUTPUT, 0.0, 100.0, cfg.RES_RISK),
    ]:
        sub = validate_mf_set(mf_set, lo, hi, step, name=name)
        report.checks[name] = sub
        if not sub["ok"]:
            report.ok = False
            report.errors.extend(f"[{name}] {e}" for e in sub["errors"])

    return report


def validate_rule_coverage(d_step: float = 0.02, lag_step: float = 0.05
                           ) -> ValidationReport:
    """Phủ luật 2 chiều: mọi (d, LAG) phải kích hoạt ít nhất một luật."""
    report = ValidationReport(name="Kiểm tra phủ luật 2 chiều", ok=True)
    fis = DomainFIS("GTNN")
    gaps = []
    for d in make_universe(0.0, 1.0, d_step):
        for lag in make_universe(0.0, cfg.LAG_MAX, lag_step):
            total = sum(w for _, w, _ in fis._firing(float(d), float(lag)))
            if total <= 1e-12:
                gaps.append((round(float(d), 3), round(float(lag), 3)))
    report.checks["n_gaps"] = len(gaps)
    report.checks["examples"] = gaps[:5]
    if gaps:
        report.ok = False
        report.errors.append(
            f"{len(gaps)} điểm (d, LAG) không kích hoạt luật nào, ví dụ {gaps[:5]}")
    return report


def output_surface(domain: str = "GTNN",
                   n_d: int = 101, n_lag: int = 101) -> tuple:
    """Mặt đầu ra risk(d, LAG) — dùng vẽ hình §9.6.

    Tiêu chí nghiệm thu B2: đường cong phải TRƠN, không phải hàm bậc thang.
    """
    fis = DomainFIS(domain)
    ds = np.linspace(0.0, 1.0, n_d)
    lags = np.linspace(0.0, cfg.LAG_MAX, n_lag)
    grid = np.array([[fis.infer(float(d), float(l)) for l in lags] for d in ds])
    return ds, lags, grid


def max_step(domain: str = "GTNN", lag: float = 1.0, n: int = 1001) -> float:
    """Bước nhảy lớn nhất của risk khi d biến thiên — chỉ số định lượng độ trơn."""
    fis = DomainFIS(domain)
    values = [fis.infer(float(d), lag) for d in np.linspace(0.0, 1.0, n)]
    return float(max(abs(b - a) for a, b in zip(values, values[1:])))
