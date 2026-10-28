# -*- coding: utf-8 -*-
"""
ts_fusion.py — §7 TẦNG HỢP NHẤT VÀ ĐẦU RA MỜ.

Đây là phần trả lời trực tiếp cho yêu cầu "đầu ra là fuzzy": kết quả cuối cùng
KHÔNG phải một nhãn mà là một vector độ thuộc, kèm khoảng bất định, hồ sơ 6 lĩnh
vực, điểm từng nhánh, và xác suất hậu nghiệm đã hiệu chỉnh.

GUI chỉ được gọi `screen()` (§10.1).
"""

from __future__ import annotations

from typing import Optional, Sequence

import config as cfg
from ts_aggregate import aggregate_rule_branch, renormalize_for_missing
from ts_dataset import validate_record
from ts_features import build_feature_vector, questions_by_domain
from ts_fuzzy import build_all_fis, infer_all_domains, risk_to_unit
from ts_mf import membership
from ts_types import (FeatureVector, Question, Record, SafetyNetViolation,
                      ScreeningResult, TrendResult, risk_bit)


# =====================================================================
# §7.4 — mờ hoá đầu ra
# =====================================================================

def fuzzify_output(score: float) -> dict[str, float]:
    """→ {'BINH_THUONG': .05, 'CAN_THEO_DOI': .22, 'NGHI_NGO': .83}

    Vùng chồng lấn [35, 42] và [58, 68] là chồng lấn THẬT, không phải danh nghĩa.
    """
    return {level: membership(cfg.MF_OUTPUT[level], score)
            for level in cfg.OUTPUT_LEVEL_ORDER}


def level_of(score: float) -> str:
    mu = fuzzify_output(score)
    return max(cfg.OUTPUT_LEVEL_ORDER, key=lambda k: (mu[k], cfg.OUTPUT_LEVEL_ORDER.index(k)))


def confidence_of(memberships: dict[str, float]) -> float:
    """§9.5 — μ_max − μ_second.

    Đây là đại lượng mà phương pháp cộng tổng điểm không có: nó cho phép mô
    hình tự nhận biết hồ sơ nào nằm sát ranh giới.
    """
    ordered = sorted(memberships.values(), reverse=True)
    if len(ordered) < 2:
        return float(ordered[0]) if ordered else 0.0
    return float(ordered[0] - ordered[1])


def crossover_point(level_a: str, level_b: str,
                    lo: float = 0.0, hi: float = 100.0,
                    tol: float = 1e-9) -> float:
    """Điểm giao μ_a(s) = μ_b(s) — tìm bằng chia đôi.

    §7.2 yêu cầu tính lại điểm này mỗi khi MF_OUTPUT thay đổi, và cài phép kiểm
    tra đó vào test thay vì tin vào con số cứng.
    """
    def gap(s: float) -> float:
        return membership(cfg.MF_OUTPUT[level_a], s) - membership(cfg.MF_OUTPUT[level_b], s)

    # Hai hàm thuộc cùng bằng 0 trên những đoạn dài ở hai đầu universe, nên
    # gap = 0 ở đó và phép chia đôi thẳng từ [0, 100] sẽ "bắt" nhầm s = 0.
    # Quét lưới thô trước để tìm đoạn ĐỔI DẤU THẬT, rồi mới chia đôi.
    step = 0.01
    n = int(round((hi - lo) / step))
    prev_x = prev_gap = None
    bracket = None
    for i in range(n + 1):
        x = lo + i * step
        g = gap(x)
        if abs(g) < 1e-15:
            continue
        if prev_gap is not None and g * prev_gap < 0:
            bracket = (prev_x, x)
            break
        prev_x, prev_gap = x, g

    if bracket is None:
        raise ValueError(
            f"Không có điểm giao giữa {level_a} và {level_b} trên [{lo}, {hi}]")

    a, b = bracket
    ga = gap(a)
    while b - a > tol:
        mid = (a + b) / 2.0
        if ga * gap(mid) <= 0:
            b = mid
        else:
            a, ga = mid, gap(mid)
    return (a + b) / 2.0


# =====================================================================
# §7.2 — LƯỚI AN TOÀN. RÀNG BUỘC TUYỆT ĐỐI.
# =====================================================================

def domain_risk_counts(record: Record,
                       questions: Sequence[Question]) -> dict[str, tuple[int, int]]:
    """{lĩnh vực: (số câu mang nguy cơ, số câu đã trả lời)} — số ĐẾM THÔ.

    Lưới an toàn là một ngưỡng pháp lý áp trên câu trả lời gốc, không phải trên
    d_i đã mờ hoá, nên hàm này cố tình không đi qua tầng đặc trưng.
    """
    if record.domain_counts is not None:
        return {d: (int(c["risk"]), int(c["n"])) for d, c in record.domain_counts.items()}

    out: dict[str, tuple[int, int]] = {}
    for domain, qs in questions_by_domain(questions).items():
        n_risk = n_ans = 0
        for q in qs:
            if not q.applies_to(record.age_days):
                continue
            bit = risk_bit((record.answers or {}).get(q.question_id), q.risk_answer)
            if bit is None:
                continue
            n_ans += 1
            n_risk += int(bit)
        if n_ans:
            out[domain] = (n_risk, n_ans)
    return out


def byt_threshold_met(record: Record,
                      questions: Sequence[Question]) -> tuple[bool, list[str]]:
    """Ngưỡng chính thức theo QĐ 2254/QĐ-BYT (§7.2).

      • Trẻ < 28 ngày (Phiếu 1): có ≥ 1 câu trả lời mang nguy cơ
      • Trẻ ≥ 28 ngày (Phiếu 2): có ≥ 1 lĩnh vực chính đạt ≥ 2/3 câu "Không"
      • Mọi độ tuổi           : có ≥ 1 dấu hiệu ở mục "Dấu hiệu bất thường khác"

    Trả (đã kích hoạt, danh sách lý do).
    """
    counts = domain_risk_counts(record, questions)
    reasons: list[str] = []

    if record.age_days < cfg.BYT_NEWBORN_AGE_DAYS:
        total_risk = sum(r for r, _ in counts.values())
        if total_risk >= cfg.BYT_NEWBORN_MIN_RISK_ITEMS:
            reasons.append(f"Phiếu 1 (< {cfg.BYT_NEWBORN_AGE_DAYS} ngày): "
                           f"{total_risk} câu trả lời mang nguy cơ")
        return bool(reasons), reasons

    for domain in cfg.MAIN_DOMAINS:
        if domain not in counts:
            continue
        n_risk, n_ans = counts[domain]
        if n_ans and n_risk / n_ans >= cfg.BYT_DOMAIN_RATIO:
            reasons.append(f"{cfg.DOMAIN_NAMES[domain]}: {n_risk}/{n_ans} câu "
                           f'"Không" (≥ 2/3)')

    if "DHBT" in counts:
        n_risk, _ = counts["DHBT"]
        if n_risk >= cfg.BYT_DHBT_MIN_SIGNS:
            reasons.append(f"{cfg.DOMAIN_NAMES['DHBT']}: {n_risk} dấu hiệu")

    return bool(reasons), reasons


def apply_safety_net(score: float,
                     record: Record,
                     questions: Sequence[Question]) -> tuple[float, bool, list[str]]:
    """Học máy được phép NÂNG mức cảnh báo, không được phép HẠ.

    Một công cụ sàng lọc y tế không được hạ mức cảnh báo xuống dưới ngưỡng pháp
    lý chỉ vì mô hình học máy "nghĩ khác".
    """
    triggered, reasons = byt_threshold_met(record, questions)
    if triggered:
        return max(score, cfg.SAFETY_FLOOR), True, reasons
    return score, False, reasons


# =====================================================================
# §7.1 — trộn ba nhánh
# =====================================================================

def fuse(branch_scores: dict[str, Optional[float]],
         weights: dict[str, float] = None) -> tuple[float, dict[str, float]]:
    """[REVIEW-P7.7] Chuẩn hoá lại trọng số trên tập nhánh KHẢ DỤNG.

    Mã giả Phụ lục B truyền thẳng `s2 = None` vào `fuse()`. Ở đây nhánh None bị
    loại và trọng số chia lại cho các nhánh còn lại — cùng thủ pháp với CH-5.
    Thà nói rõ một nhánh bị tắt còn hơn báo một con số không đáng tin (§7.1).
    """
    weights = weights or cfg.FUSION_WEIGHTS
    usable = {k: v for k, v in branch_scores.items() if v is not None}
    if not usable:
        raise ValueError("Không nhánh nào cho được điểm")

    w_sum = sum(weights[k] for k in usable)
    if w_sum <= 0:
        raise ValueError("Tổng trọng số của các nhánh khả dụng không dương")
    used = {k: weights[k] / w_sum for k in usable}
    return sum(used[k] * usable[k] for k in usable), used


# =====================================================================
# §7.3 — khoảng bất định
# =====================================================================

def uncertainty_interval(score: float,
                         reliability: float,
                         branch_scores: dict[str, Optional[float]],
                         ambiguity_half_width: float = 0.0) -> tuple[float, float]:
    """half_width = (1 − rel)·W_MAX + β·spread + γ·ambiguity

    Ba nguồn bất định tách bạch:
      • (1 − rel)·W_MAX  — chất lượng phiếu trả lời (§2.4)
      • β·spread         — ba nhánh bất đồng đến đâu
      • γ·ambiguity      — [REVIEW-P3] không biết câu nào chưa đạt (dữ liệu gộp);
                           bằng 0 khi có dữ liệu từng câu

    Đây là cách biến hạn chế "phụ thuộc chất lượng dữ liệu khảo sát" thành một
    TÍNH NĂNG của hệ thống, và là bước đệm tự nhiên tới tập mờ loại 2 khoảng.
    """
    usable = [v for v in branch_scores.values() if v is not None]
    spread = (max(usable) - min(usable)) if len(usable) > 1 else 0.0
    half = ((1.0 - reliability) * cfg.UNC_W_MAX
            + cfg.UNC_BETA * spread
            + cfg.UNC_GAMMA * ambiguity_half_width)
    return (max(0.0, score - half), min(100.0, score + half))


def interval_advisory(score: float, interval: tuple[float, float],
                      reliability: float) -> list[str]:
    """Diễn giải theo khoảng — bảng ở §7.3."""
    notes: list[str] = []
    if level_of(interval[0]) != level_of(interval[1]):
        notes.append("Kết quả chưa đủ tin cậy — đề nghị đánh giá lại sau 4 tuần "
                     "(khoảng bất định phủ từ hai mức trở lên).")
    if reliability < cfg.REL_WARN_THRESHOLD:
        notes.append(f"Chất lượng phiếu thấp (rel = {reliability:.2f}) — đề nghị "
                     "người khác điền lại.")
    return notes


# =====================================================================
# §7.6 — hiệu chỉnh sang xác suất hậu nghiệm
# =====================================================================

def posterior_probability(positive: bool, se: float, sp: float,
                          prevalence: float) -> float:
    """P(RLPTK | kết quả) bằng Bayes với tỷ lệ hiện mắc.

    Điểm 80/100 không có ý nghĩa lâm sàng trực tiếp; con số này thì có.
    Tránh đồng thời hai lỗi: gây hoang mang cho phụ huynh, và tạo cảm giác
    chính xác giả từ điểm số.
    """
    se = min(max(se, 1e-6), 1 - 1e-6)
    sp = min(max(sp, 1e-6), 1 - 1e-6)
    lr = se / (1 - sp) if positive else (1 - se) / sp
    odds_pre = prevalence / (1 - prevalence)
    odds_post = odds_pre * lr
    return odds_post / (1 + odds_post)


def posterior_all(positive: bool, se: float, sp: float) -> dict[str, float]:
    return {name: posterior_probability(positive, se, sp, p)
            for name, p in cfg.PREVALENCE.items()}


# =====================================================================
# §7.5 — phân tích diễn tiến theo thời gian
# =====================================================================

def analyze_trend(history: Sequence[tuple[str, float]]) -> Optional[TrendResult]:
    """history: [(assessment_date ISO, score), ...] theo thứ tự thời gian.

    Ba mốc sàng lọc theo tuổi (ví dụ 12 – 18 – 24 tháng) tạo thành một chuỗi
    thật có thứ tự thời gian — sau bước này cụm từ "phân tích chuỗi hành vi"
    mô tả đúng cái mô hình làm.
    """
    import datetime as _dt

    if len(history) < 2:
        return None
    (d0, s0), (d1, s1) = history[-2], history[-1]
    delta_days = (_dt.date.fromisoformat(d1) - _dt.date.fromisoformat(d0)).days
    if delta_days <= 0:
        raise ValueError("Hai lần đánh giá phải cách nhau ít nhất một ngày")

    delta = s1 - s0
    rate = delta / (delta_days / 30.0)
    mu = {name: membership(spec, rate) for name, spec in cfg.MF_TREND.items()}
    trend = max(mu, key=mu.get)
    current = level_of(s1)

    priority = "TIEP_TUC_THEO_DOI"
    for rule_trend, rule_level, rule_priority in cfg.RULES_TREND:
        if rule_trend == trend and rule_level == current:
            priority = rule_priority
            break

    return TrendResult(delta_score=delta, delta_days=delta_days,
                       rate_per_month=rate, trend_memberships=mu,
                       trend_label=trend, priority=priority,
                       n_assessments=len(history))


# =====================================================================
# Hàm điều phối duy nhất (§7.7)
# =====================================================================

def record_to_x(record: Record,
                questions: Sequence[Question],
                fis: dict = None,
                which: str = "point") -> dict[str, float]:
    """Record → vector 6 chiều x_i = risk_i / 100 ∈ [0,1] cho các nhánh §4–§6.

    Tách riêng khỏi `screen()` để nhánh học máy dùng lại được khi huấn luyện.
    `which` ∈ {"lo", "point", "hi"} để lan truyền khoảng.
    """
    features = build_feature_vector(record, questions)
    risk = infer_all_domains(features, fis or build_all_fis())
    return risk_to_unit({d: iv for d, iv in risk.items() if iv is not None}, which)


class Models:
    """Gói mô hình đã huấn luyện. Nhánh chưa sẵn sàng thì để None."""

    def __init__(self, fis=None, fknn=None, fcm=None, se: float = 0.9, sp: float = 0.7):
        self.fis = fis or build_all_fis()
        self.fknn = fknn
        self.fcm = fcm
        self.se = se
        self.sp = sp


def screen(record: Record,
           questions: Sequence[Question],
           models: Models = None,
           aggregate_operator: str = "choquet") -> ScreeningResult:
    """Hàm điều phối duy nhất — GUI chỉ gọi hàm này."""
    models = models or Models()
    warnings = validate_record(record, questions)          # INV-1..INV-5

    # --- §2: đặc trưng ---
    features: FeatureVector = build_feature_vector(record, questions)

    # --- §3: FIS lớp 1 ---
    risk = infer_all_domains(features, models.fis)
    risk = {d: iv for d, iv in risk.items() if iv is not None}
    if not risk:
        raise ValueError(f"[{record.record_id}] không lĩnh vực nào chấm được")

    # --- §4: nhánh luật, lan truyền cả khoảng ---
    s1 = aggregate_rule_branch(risk_to_unit(risk, "point"), operator=aggregate_operator)
    s1_lo = aggregate_rule_branch(risk_to_unit(risk, "lo"), operator=aggregate_operator)
    s1_hi = aggregate_rule_branch(risk_to_unit(risk, "hi"), operator=aggregate_operator)

    # --- §5, §6: hai nhánh học máy ---
    x = risk_to_unit(risk, "point")
    s2 = models.fknn.score(x) if models.fknn is not None else None
    s3 = models.fcm.score(x) if models.fcm is not None else None

    # --- §7.1: hợp nhất ---
    branch_scores: dict[str, Optional[float]] = {"rule": s1, "fknn": s2, "fcm": s3}
    score, weights_used = fuse(branch_scores)

    # --- §7.2: LƯỚI AN TOÀN — KHÔNG BỎ QUA ---
    score, safety, reasons = apply_safety_net(score, record, questions)

    # --- §7.3: khoảng bất định ---
    ambiguity = cfg.UNC_GAMMA * (s1_hi - s1_lo) / 2.0
    lo, hi = uncertainty_interval(score, features.reliability, branch_scores,
                                  ambiguity_half_width=ambiguity)

    # --- §7.4: vector độ thuộc ---
    mu = fuzzify_output(score)
    level = level_of(score)
    confidence = confidence_of(mu)

    # --- §7.6: xác suất hậu nghiệm ---
    posterior = posterior_all(level == "NGHI_NGO", models.se, models.sp)

    advisory = list(warnings) + interval_advisory(score, (lo, hi), features.reliability)
    if safety:
        advisory.append(cfg.SAFETY_NET_NOTICE + " Lý do: " + "; ".join(reasons) + ".")
    if features.excluded_domains:
        advisory.append("Lĩnh vực bị loại khỏi tổng hợp: "
                        + ", ".join(cfg.DOMAIN_NAMES[d] for d in features.excluded_domains)
                        + " (trọng số đã chuẩn hoá lại theo CH-5).")
    if not features.itemwise:
        advisory.append(
            "Hồ sơ nhập ở mức lĩnh vực nên chỉ số trễ tuổi được suy ra dưới dạng "
            f"khoảng; điểm nhánh luật nằm trong [{s1_lo:.1f}; {s1_hi:.1f}].")

    rule_trace = []
    for domain, feat in features.domains.items():
        if feat.available and feat.d is not None:
            rule_trace.extend(models.fis[domain].explain(feat.d.point,
                                                         feat.lag.point if feat.lag else 0.0))

    result = ScreeningResult(
        score=score,
        memberships=mu,
        level=level,
        confidence=confidence,
        interval=(lo, hi),
        domain_profile={d: iv.point for d, iv in risk.items()},
        branch_scores=branch_scores,
        safety_triggered=safety,
        posterior_prob=posterior,
        cluster_profile=models.fcm.describe_nearest(x) if models.fcm is not None else "",
        neighbors=models.fknn.explain(x) if models.fknn is not None else [],
        rule_trace=rule_trace,
        reliability=features.reliability,
        score_bounds=(s1_lo, s1_hi),
        domain_profile_bounds={d: (iv.lo, iv.hi) for d, iv in risk.items()},
        age_group=features.age_group,
        excluded_domains=features.excluded_domains,
        fusion_weights_used=weights_used,
        advisory=advisory,
    )

    _check_invariants(result)
    return result


def _check_invariants(result: ScreeningResult) -> None:
    """§12.3 — bất biến an toàn.

    [REVIEW-P7.8] Dùng ngoại lệ thật chứ không phải `assert`: `python -O` xoá
    sạch mọi assert, mà ràng buộc này được mô tả là "không được phép bỏ qua
    trong bất kỳ phiên bản nào".
    """
    if result.safety_triggered and result.level != "NGHI_NGO":
        raise SafetyNetViolation(
            f"VI PHẠM LƯỚI AN TOÀN: hồ sơ chạm ngưỡng BYT (điểm {result.score:.2f}) "
            f"nhưng kết luận là {result.level}. Kiểm tra SAFETY_FLOOR so với điểm "
            "giao CAN_THEO_DOI/NGHI_NGO của MF_OUTPUT.")
    total_mu = sum(result.memberships.values())
    if not (0.0 <= total_mu <= 3.0):
        raise SafetyNetViolation(f"Tổng độ thuộc bất thường: {total_mu}")
    lo, hi = result.interval
    if not (lo <= result.score <= hi):
        raise SafetyNetViolation(
            f"Điểm {result.score} nằm ngoài khoảng bất định [{lo}, {hi}]")
