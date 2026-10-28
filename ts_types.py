# -*- coding: utf-8 -*-
"""
ts_types.py — Kiểu dữ liệu dùng chung cho toàn hệ thống.

Tham chiếu: §1.1, §1.3, §2.6, §7.4.
Phần mở rộng khoảng (lo/point/hi) là hệ quả của [REVIEW-P3]: khi dữ liệu chỉ
ghi ở mức lĩnh vực, mọi đại lượng dẫn xuất đều là KHOẢNG chứ không phải điểm.
Khi có dữ liệu từng câu, lo == point == hi và mọi công thức phía sau không đổi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# =====================================================================
# Ngân hàng câu hỏi (§1.1)
# =====================================================================

@dataclass(frozen=True)
class Question:
    question_id: str
    domain: str
    question_content: str
    risk_answer: str            # "KHONG" hoặc "CO"
    expected_age_days: int      # a_j
    tolerance_days: int         # σ_j
    weight: float               # w_j
    is_critical: int
    age_min_days: int
    age_max_days: int
    abstract_level: float
    source: str = ""

    def applies_to(self, age_days: int) -> bool:
        """INV-3 — câu hỏi chỉ được tính trong cửa sổ tuổi của nó."""
        return self.age_min_days <= age_days <= self.age_max_days


def risk_bit(answer: Optional[str], risk_answer: str) -> Optional[float]:
    """Chuẩn hoá câu trả lời về 'bit nguy cơ' (§1.1).

    Trả None khi bỏ trống — xử lý riêng ở §2.5, tuyệt đối KHÔNG điền 0.
    """
    if answer is None:
        return None
    return 1.0 if str(answer).strip().upper() == risk_answer else 0.0


# =====================================================================
# Hồ sơ sàng lọc (§1.3)
# =====================================================================

@dataclass
class Record:
    record_id: str
    child_id: str
    assessment_date: str
    birth_date: str
    age_days: int

    # Một trong hai dạng dưới đây phải có mặt.
    answers: Optional[dict[str, Optional[str]]] = None
    # {"GTNN": {"n": 3, "risk": 2}, ...} — dạng gộp của dữ liệu thật hiện có.
    domain_counts: Optional[dict[str, dict[str, int]]] = None

    duration_seconds: Optional[int] = None
    informant: Optional[str] = None
    label: Optional[int] = None
    label_source: Optional[str] = None
    data_origin: str = "real"
    notes: str = ""

    @property
    def is_itemwise(self) -> bool:
        """True nếu hồ sơ có đáp án từng câu (đặc trưng là điểm, không phải khoảng)."""
        return self.answers is not None

    @property
    def age_group(self) -> str:
        """[REVIEW-P4] nhãn nhóm tuổi phục vụ báo cáo tách nhóm ở §9."""
        return "trong_pham_vi" if self.age_days <= 1095 else "ngoai_pham_vi"


# =====================================================================
# Đặc trưng (§2)
# =====================================================================

@dataclass
class Interval:
    """Khoảng [lo, hi] kèm một ước lượng điểm nằm trong khoảng."""
    lo: float
    point: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def is_exact(self) -> bool:
        return self.width <= 1e-9

    @staticmethod
    def exact(value: float) -> "Interval":
        return Interval(value, value, value)

    def __post_init__(self) -> None:
        if not (self.lo - 1e-9 <= self.point <= self.hi + 1e-9):
            raise ValueError(
                f"Ước lượng điểm {self.point} nằm ngoài khoảng [{self.lo}, {self.hi}]")

    def __repr__(self) -> str:  # pragma: no cover - chỉ phục vụ hiển thị
        if self.is_exact:
            return f"{self.point:.4g}"
        return f"{self.point:.4g} [{self.lo:.4g}, {self.hi:.4g}]"


@dataclass
class DomainFeature:
    """Đặc trưng của một lĩnh vực. `available = False` khi lĩnh vực bị loại (§2.5)."""
    domain: str
    available: bool
    d: Optional[Interval] = None            # §2.1 — điểm thiếu hụt
    lag: Optional[Interval] = None          # §2.3 — chỉ số trễ tuổi
    n_applicable: int = 0
    n_answered: int = 0
    n_risk: Optional[int] = None
    miss_rate: float = 0.0
    exclusion_reason: str = ""

    # Tập (d, lag) khả dĩ khi dữ liệu ở mức lĩnh vực. Với dữ liệu từng câu,
    # danh sách có đúng một phần tử. FIS lớp 1 chạy trên toàn bộ danh sách này
    # để lấy khoảng risk chính xác thay vì dùng số học khoảng.
    combos: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_exact(self) -> bool:
        return len(self.combos) <= 1


@dataclass
class FeatureVector:
    """Đầu vào duy nhất của cả ba nhánh (§2.6)."""
    record_id: str
    age_days: int
    age_group: str
    domains: dict[str, DomainFeature]
    reliability: float
    miss_rate_all: float = 0.0
    inconsistency: Optional[float] = None    # None khi không tính được (dữ liệu gộp)
    speed_flag: int = 0
    itemwise: bool = True

    @property
    def available_domains(self) -> list[str]:
        return [k for k, v in self.domains.items() if v.available]

    @property
    def excluded_domains(self) -> list[str]:
        return [k for k, v in self.domains.items() if not v.available]


# =====================================================================
# Giải thích (§3.4, §5.5, §6.5)
# =====================================================================

@dataclass
class RuleActivation:
    domain: str
    rule_name: str
    antecedent: str
    firing_strength: float
    consequent: str


@dataclass
class Neighbor:
    record_id: str
    distance: float
    label: int
    label_text: str


# =====================================================================
# Kết quả sàng lọc (§7.4)
# =====================================================================

@dataclass
class ScreeningResult:
    score: float                          # điểm giải mờ 0–100
    memberships: dict[str, float]         # vector độ thuộc trên 3 mức
    level: str                            # argmax của memberships
    confidence: float                     # μ_max − μ_second
    interval: tuple[float, float]         # khoảng bất định §7.3
    domain_profile: dict[str, Optional[float]]
    branch_scores: dict[str, Optional[float]]
    safety_triggered: bool
    posterior_prob: dict[str, float]
    cluster_profile: str
    neighbors: list[Neighbor]
    rule_trace: list[RuleActivation]
    reliability: float

    # Mở rộng [REVIEW-P3] / [REVIEW-P4] — không thay thế trường nào ở trên.
    score_bounds: tuple[float, float] = (0.0, 100.0)   # khoảng do dữ liệu gộp
    domain_profile_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    age_group: str = "trong_pham_vi"
    excluded_domains: list[str] = field(default_factory=list)
    fusion_weights_used: dict[str, float] = field(default_factory=dict)
    advisory: list[str] = field(default_factory=list)

    @property
    def interval_spans_multiple_levels(self) -> bool:
        """§7.3 — khoảng phủ ≥ 2 mức thì kết quả chưa đủ tin cậy."""
        from ts_fusion import level_of  # nhập trễ để tránh phụ thuộc vòng
        return level_of(self.interval[0]) != level_of(self.interval[1])


@dataclass
class TrendResult:
    """§7.5 — diễn tiến theo thời gian."""
    delta_score: float
    delta_days: int
    rate_per_month: float
    trend_memberships: dict[str, float]
    trend_label: str
    priority: str
    n_assessments: int


# =====================================================================
# Báo cáo kiểm tra
# =====================================================================

@dataclass
class ValidationReport:
    name: str
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(f"{self.name} thất bại:\n  - " + "\n  - ".join(self.errors))


class SafetyNetViolation(RuntimeError):
    """[REVIEW-P7.8] §12.3 dùng `assert`, mà `python -O` xoá sạch assert.

    Một ràng buộc được mô tả là "không được phép bỏ qua trong bất kỳ phiên bản
    nào" thì phải là ngoại lệ thật, không phải assert.
    """
