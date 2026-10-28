# -*- coding: utf-8 -*-
"""
ts_features.py — §2 TẦNG ĐẶC TRƯNG.

Biến 20 bit nhị phân thành các đại lượng liên tục thực sự mờ được:
    d_i    điểm thiếu hụt lĩnh vực          ∈ [0, 1]
    LAG_i  chỉ số trễ phát triển theo tuổi  ∈ [0, LAG_MAX]
    rel    độ tin cậy phiếu trả lời         ∈ [0, 1]

[REVIEW-P3] — MỞ RỘNG QUAN TRỌNG SO VỚI ĐẶC TẢ v1.0
-----------------------------------------------------
Đặc tả giả định biết đáp án TỪNG CÂU. Dữ liệu thật hiện có chỉ ghi "số câu Có /
số câu Không" theo lĩnh vực, nên với một lĩnh vực có k trong n câu mang nguy cơ
ta không biết là những câu NÀO — mà a_j khác nhau giữa các câu, nên LAG_i không
xác định duy nhất.

Cách xử lý ở đây: liệt kê VÉT CẠN mọi cấu hình khả dĩ (n ≤ 5 nên tối đa vài
chục cấu hình mỗi lĩnh vực) và giữ lại toàn bộ tập (d, LAG) đạt được. FIS lớp 1
chạy trên cả tập đó để lấy [risk_lo, risk_hi] CHÍNH XÁC.

Vì sao không dùng số học khoảng: d_i và LAG_i cùng phụ thuộc vào một lựa chọn
tập con, nên nhân/cộng hai khoảng độc lập sẽ cho khoảng RỘNG HƠN thực tế
("bài toán phụ thuộc"). Liệt kê vét cạn không mắc lỗi này và ở kích thước n ≤ 5
thì rẻ hơn mọi phương án xấp xỉ.

Khi có dữ liệu từng câu, mỗi lĩnh vực chỉ có một cấu hình → khoảng suy biến
thành điểm và mọi công thức phía sau trùng khớp đặc tả gốc.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Optional, Sequence

import config as cfg
from ts_types import DomainFeature, FeatureVector, Interval, Question, Record, risk_bit


class FeatureError(ValueError):
    """Hồ sơ không đủ điều kiện chấm điểm (INV-4)."""


# =====================================================================
# Tiện ích
# =====================================================================

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def questions_by_domain(questions: Sequence[Question]) -> dict[str, list[Question]]:
    out: dict[str, list[Question]] = {d: [] for d in cfg.DOMAINS}
    for q in questions:
        out.setdefault(q.domain, []).append(q)
    return out


def potential_lag(question: Question, age_days: int) -> float:
    """lag_j nếu câu j mang nguy cơ (§2.3).

    Trẻ 320 ngày chưa đạt mốc kỳ vọng ở 300 ngày → lag ≈ 0,22 (bình thường);
    trẻ 700 ngày chưa đạt chính mốc đó → lag = 3,0 (đã chặn trên, rất nặng).
    Mô hình cộng điểm coi hai ca này giống hệt nhau.
    """
    if question.domain in cfg.AGELESS_DOMAINS:
        return 0.0
    tol = max(1, int(question.tolerance_days))
    return _clip((age_days - question.expected_age_days) / tol, 0.0, cfg.LAG_MAX)


def _point_estimate(values: Sequence[float]) -> float:
    """Ước lượng điểm từ tập giá trị khả dĩ, theo cfg.AGG_POINT_ESTIMATE."""
    if not values:
        raise FeatureError("Không có cấu hình khả dĩ nào để ước lượng")
    if cfg.AGG_POINT_ESTIMATE == "mid":
        return (min(values) + max(values)) / 2.0
    if cfg.AGG_POINT_ESTIMATE == "mean":
        return sum(values) / len(values)
    raise FeatureError(f"AGG_POINT_ESTIMATE không hợp lệ: {cfg.AGG_POINT_ESTIMATE!r}")


def _interval_from(values: Sequence[float]) -> Interval:
    lo, hi = min(values), max(values)
    return Interval(lo, _clip(_point_estimate(values), lo, hi), hi)


# =====================================================================
# Đặc trưng lĩnh vực — dữ liệu TỪNG CÂU
# =====================================================================

def _domain_feature_itemwise(domain: str,
                             qs: Sequence[Question],
                             answers: dict[str, Optional[str]],
                             age_days: int) -> DomainFeature:
    applicable = [q for q in qs if q.applies_to(age_days)]
    n_app = len(applicable)
    if n_app == 0:
        return DomainFeature(domain=domain, available=False, n_applicable=0,
                             exclusion_reason="không có câu hỏi nào áp dụng cho độ tuổi này")

    answered: list[tuple[Question, float]] = []
    for q in applicable:
        bit = risk_bit(answers.get(q.question_id), q.risk_answer)
        if bit is not None:
            answered.append((q, bit))

    miss_rate = 1.0 - len(answered) / n_app
    if miss_rate > cfg.MISS_THRESHOLD_DOM or not answered:
        # §2.5 — CẤM điền khuyết bằng 0: đó là giả định ngầm "trẻ đạt mốc",
        # làm giảm độ nhạy một cách nguy hiểm trong bối cảnh sàng lọc.
        return DomainFeature(domain=domain, available=False,
                             n_applicable=n_app, n_answered=len(answered),
                             miss_rate=miss_rate,
                             exclusion_reason=f"bỏ trống {miss_rate:.0%} > "
                                              f"{cfg.MISS_THRESHOLD_DOM:.0%}")

    w_total = sum(q.weight for q, _ in answered)
    d_value = sum(q.weight * bit for q, bit in answered) / w_total

    lags = [potential_lag(q, age_days) for q, bit in answered if bit == 1.0]
    # `max` chứ không phải `mean`: một mốc trễ nặng không được phép bị các mốc
    # đã đạt "pha loãng" (§2.3).
    lag_value = max(lags) if lags else 0.0

    return DomainFeature(
        domain=domain, available=True,
        d=Interval.exact(d_value), lag=Interval.exact(lag_value),
        n_applicable=n_app, n_answered=len(answered),
        n_risk=int(sum(bit for _, bit in answered)),
        miss_rate=miss_rate,
        combos=[(d_value, lag_value)],
    )


# =====================================================================
# Đặc trưng lĩnh vực — dữ liệu MỨC LĨNH VỰC (liệt kê vét cạn)
# =====================================================================

MAX_COMBOS = 512    # chốt chặn: n ≤ 5 thì thực tế không bao giờ chạm tới


def _enumerate_combos(applicable: Sequence[Question],
                      n_answered: int,
                      n_risk: int,
                      age_days: int) -> list[tuple[float, float]]:
    """Mọi cặp (d, LAG) đạt được khi biết n_answered câu được trả lời và
    n_risk trong số đó mang nguy cơ, nhưng không biết là những câu nào."""
    idx = range(len(applicable))
    combos: set[tuple[float, float]] = set()

    for answered_idx in itertools.combinations(idx, n_answered):
        answered_qs = [applicable[i] for i in answered_idx]
        w_total = sum(q.weight for q in answered_qs)
        if w_total <= 0:
            continue
        for risk_idx in itertools.combinations(range(n_answered), n_risk):
            risk_qs = [answered_qs[i] for i in risk_idx]
            d_value = sum(q.weight for q in risk_qs) / w_total
            lags = [potential_lag(q, age_days) for q in risk_qs]
            lag_value = max(lags) if lags else 0.0
            combos.add((round(d_value, 12), round(lag_value, 12)))
            if len(combos) > MAX_COMBOS:
                raise FeatureError(
                    f"Số cấu hình vượt {MAX_COMBOS}; kiểm tra lại ngân hàng câu hỏi")

    return sorted(combos)


def _domain_feature_aggregated(domain: str,
                               qs: Sequence[Question],
                               n_answered: int,
                               n_risk: int,
                               age_days: int) -> DomainFeature:
    if cfg.AGG_ASSUME_ALL_APPLICABLE:
        # Phiếu được phát nguyên bộ, nên bỏ lọc tuổi ở mức câu; phạm vi tuổi
        # được kiểm ở mức hồ sơ thay vì mức câu.
        applicable = list(qs)
    else:
        applicable = [q for q in qs if q.applies_to(age_days)]

    n_app = len(applicable)
    if n_app == 0:
        return DomainFeature(domain=domain, available=False, n_applicable=0,
                             exclusion_reason="không có câu hỏi nào áp dụng cho độ tuổi này")
    if n_answered > n_app:
        raise FeatureError(
            f"[{domain}] số câu đã trả lời ({n_answered}) vượt số câu của lĩnh vực ({n_app})")
    if not (0 <= n_risk <= n_answered):
        raise FeatureError(f"[{domain}] số câu mang nguy cơ không hợp lệ: {n_risk}/{n_answered}")

    miss_rate = 1.0 - n_answered / n_app
    if miss_rate > cfg.MISS_THRESHOLD_DOM or n_answered == 0:
        return DomainFeature(domain=domain, available=False,
                             n_applicable=n_app, n_answered=n_answered,
                             miss_rate=miss_rate,
                             exclusion_reason=f"bỏ trống {miss_rate:.0%} > "
                                              f"{cfg.MISS_THRESHOLD_DOM:.0%}")

    combos = _enumerate_combos(applicable, n_answered, n_risk, age_days)
    d_values = [c[0] for c in combos]
    lag_values = [c[1] for c in combos]

    return DomainFeature(
        domain=domain, available=True,
        d=_interval_from(d_values), lag=_interval_from(lag_values),
        n_applicable=n_app, n_answered=n_answered, n_risk=n_risk,
        miss_rate=miss_rate, combos=combos,
    )


# =====================================================================
# API theo §2.6
# =====================================================================

def compute_domain_features(record: Record,
                            questions: Sequence[Question]) -> dict[str, DomainFeature]:
    """Tính đặc trưng cho cả 6 lĩnh vực, tự chọn nhánh theo dạng dữ liệu."""
    by_domain = questions_by_domain(questions)
    out: dict[str, DomainFeature] = {}

    for domain in cfg.DOMAINS:
        qs = by_domain.get(domain, [])
        if record.is_itemwise:
            out[domain] = _domain_feature_itemwise(
                domain, qs, record.answers or {}, record.age_days)
        else:
            counts = (record.domain_counts or {}).get(domain)
            if counts is None:
                out[domain] = DomainFeature(
                    domain=domain, available=False, n_applicable=len(qs),
                    exclusion_reason="lĩnh vực không có mặt trong hồ sơ")
                continue
            out[domain] = _domain_feature_aggregated(
                domain, qs, int(counts["n"]), int(counts["risk"]), record.age_days)

    return out


def compute_domain_scores(record: Record,
                          questions: Sequence[Question]) -> dict[str, Optional[float]]:
    """§2.6 — {'GTNN': 0.667, 'VDTHO': 0.0, ...}, giá trị ∈ [0,1] hoặc None."""
    feats = compute_domain_features(record, questions)
    return {k: (v.d.point if v.available and v.d else None) for k, v in feats.items()}


def compute_domain_lags(record: Record,
                        questions: Sequence[Question]) -> dict[str, Optional[float]]:
    """§2.6 — {'GTNN': 2.1, 'VDTHO': 0.0, ...}, giá trị ∈ [0, LAG_MAX] hoặc None."""
    feats = compute_domain_features(record, questions)
    return {k: (v.lag.point if v.available and v.lag else None) for k, v in feats.items()}


# =====================================================================
# §2.4 — độ tin cậy phiếu
# =====================================================================

def count_inconsistencies(answers: dict[str, Optional[str]],
                          questions: Sequence[Question],
                          age_days: int) -> tuple[int, int]:
    """Đếm (số cặp mâu thuẫn, số cặp kiểm tra được).

    Cặp (j_dễ, j_khó) cùng lĩnh vực, a_khó > a_dễ + INCONS_MIN_GAP_DAYS.
    Mâu thuẫn khi risk_khó = 0 ∧ risk_dễ = 1 — đạt mốc khó nhưng chưa đạt mốc
    dễ hơn, dấu hiệu trả lời thiếu nhất quán.
    """
    by_domain = questions_by_domain(questions)
    n_bad = n_pairs = 0

    for domain, qs in by_domain.items():
        if domain in cfg.AGELESS_DOMAINS:
            continue
        usable = [q for q in qs if q.applies_to(age_days)]
        for easy, hard in itertools.permutations(usable, 2):
            if hard.expected_age_days <= easy.expected_age_days + cfg.INCONS_MIN_GAP_DAYS:
                continue
            bit_easy = risk_bit(answers.get(easy.question_id), easy.risk_answer)
            bit_hard = risk_bit(answers.get(hard.question_id), hard.risk_answer)
            if bit_easy is None or bit_hard is None:
                continue
            n_pairs += 1
            if bit_hard == 0.0 and bit_easy == 1.0:
                n_bad += 1

    return n_bad, n_pairs


def compute_reliability(record: Record,
                        questions: Sequence[Question]) -> tuple[float, dict]:
    """§2.4 — rel ∈ [0,1] kèm chi tiết từng thành phần.

    `rel` KHÔNG tham gia suy diễn nguy cơ: phiếu kém tin cậy không có nghĩa là
    trẻ nguy cơ cao hơn. Nó chỉ điều khiển độ rộng khoảng bất định ở §7.3.

    Với hồ sơ ở mức lĩnh vực, `incons` không tính được. Thay vì gán 0 (ngầm coi
    là "hoàn toàn nhất quán"), thành phần đó bị LOẠI và trọng số được chuẩn hoá
    lại trên các thành phần còn lại — cùng tinh thần với CH-5.
    """
    by_domain = questions_by_domain(questions)
    applicable = [q for qs in by_domain.values() for q in qs if q.applies_to(record.age_days)]
    n_app = len(applicable)

    detail: dict = {"n_applicable": n_app}
    parts: list[tuple[float, float]] = []   # (trọng số, giá trị phạt)

    if record.is_itemwise:
        answers = record.answers or {}
        n_missing = sum(1 for q in applicable if answers.get(q.question_id) is None)
        miss_rate = n_missing / n_app if n_app else 1.0
        n_bad, n_pairs = count_inconsistencies(answers, questions, record.age_days)
        incons = (n_bad / n_pairs) if n_pairs else None
        detail.update(miss_rate=miss_rate, incons=incons,
                      n_incons_pairs=n_pairs, n_incons_bad=n_bad)
        parts.append((cfg.REL_W_MISS, miss_rate))
        if incons is not None:
            parts.append((cfg.REL_W_INCONS, incons))
    else:
        counts = record.domain_counts or {}
        n_answered = sum(int(c["n"]) for c in counts.values())
        miss_rate = 1.0 - n_answered / n_app if n_app else 1.0
        miss_rate = _clip(miss_rate, 0.0, 1.0)
        detail.update(miss_rate=miss_rate, incons=None,
                      incons_reason="không tính được với dữ liệu mức lĩnh vực")
        parts.append((cfg.REL_W_MISS, miss_rate))

    if record.duration_seconds is not None and n_app:
        speed_flag = 1 if record.duration_seconds < cfg.SPEED_SEC_PER_ITEM * n_app else 0
        detail["speed_flag"] = speed_flag
        parts.append((cfg.REL_W_SPEED, float(speed_flag)))
    else:
        detail["speed_flag"] = None
        detail["speed_reason"] = "thiếu duration_seconds"

    w_sum = sum(w for w, _ in parts)
    penalty = sum(w * v for w, v in parts) / w_sum if w_sum > 0 else 0.0
    rel = _clip(1.0 - penalty, 0.0, 1.0)
    detail["penalty"] = penalty
    detail["weights_used"] = w_sum
    return rel, detail


# =====================================================================
# §2.6 — hàm gộp
# =====================================================================

def build_feature_vector(record: Record,
                         questions: Sequence[Question]) -> FeatureVector:
    """Đầu vào duy nhất của cả ba nhánh. Kiểm INV-4 trước khi trả kết quả."""
    domains = compute_domain_features(record, questions)
    rel, detail = compute_reliability(record, questions)

    miss_all = float(detail.get("miss_rate") or 0.0)
    if miss_all > cfg.MISS_THRESHOLD_ALL:
        raise FeatureError(
            f"INV-4: bỏ trống {miss_all:.0%} toàn phiếu (> {cfg.MISS_THRESHOLD_ALL:.0%}) "
            "→ từ chối chấm điểm, yêu cầu điền lại")

    if not any(f.available for f in domains.values()):
        raise FeatureError(
            "Không lĩnh vực nào đủ dữ liệu để chấm điểm; kiểm tra phạm vi tuổi "
            "(INV-3) và tỷ lệ bỏ trống (§2.5)")

    return FeatureVector(
        record_id=record.record_id,
        age_days=record.age_days,
        age_group=record.age_group,
        domains=domains,
        reliability=rel,
        miss_rate_all=miss_all,
        inconsistency=detail.get("incons"),
        speed_flag=int(detail.get("speed_flag") or 0),
        itemwise=record.is_itemwise,
    )
